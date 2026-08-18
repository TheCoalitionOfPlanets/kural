"""Convert a Coqui-TTS VITS checkpoint into a Piper voice (ONNX + JSON).

Piper and Coqui both implement VITS, and for a *character-based* Coqui model
(`use_phonemes: false`) the two are close enough to convert: the generator
architecture is identical, only the module names and the text frontend differ.

    tts\\venv\\Scripts\\python.exe tts\\convert_coqui_to_piper.py \\
        --checkpoint best_model.pth --config config.json \\
        --output-dir tts/models/export/kn_syspin

This exists because Piper has no Kannada voice anywhere upstream, while the
SYSPIN project (IISc Bangalore, MIT licensed) publishes Coqui VITS checkpoints
for it. The conversion is what makes Kannada speakable at all.

Two things make it work:

* **Weights.** Coqui names the generator modules `text_encoder`, `flow`,
  `duration_predictor`, `waveform_decoder`; Piper calls the same modules
  `enc_p`, `flow`, `dp`, `dec`. The tensors are otherwise identical, so a
  name remap is enough. Coqui's `disc` (the training discriminator) is dropped.

* **Text.** Piper normally phonemizes with espeak and looks the phonemes up in
  `phoneme_id_map`. A character-based Coqui model has no phonemes — its ids
  ARE characters. So the exported voice declares `phoneme_type: "text"` and a
  phoneme_id_map built from the Coqui charset, which makes Piper pass raw
  characters through in exactly the order the embedding was trained on:

      vocab = [pad] + punctuations + characters + [blank]

  That ordering is not a guess: it is `VitsCharacters._create_vocab` in
  Coqui's `TTS/tts/models/vits.py`, and the script asserts the resulting size
  against the checkpoint's own embedding to catch any mismatch.
"""
import argparse
import json
import sys
from pathlib import Path

import torch


def build_vocab(characters_cfg):
    """Reproduce Coqui VitsCharacters._create_vocab exactly.

    VitsCharacters overrides _create_vocab, so the base class's is_sorted /
    is_unique handling never runs — the order is literally as written here.
    """
    return (
        [characters_cfg["pad"]]
        + list(characters_cfg["punctuations"])
        + list(characters_cfg["characters"])
        + [characters_cfg["blank"]]
    )


# Coqui module name -> Piper module name. Only the top-level prefix differs.
_RENAME = {
    "text_encoder.": "enc_p.",
    "posterior_encoder.": "enc_q.",
    "waveform_decoder.": "dec.",
    "duration_predictor.": "dp.",
    "flow.": "flow.",
    "emb_g.": "emb_g.",
}


def _denormalize_weight_norm(key):
    """Translate new-style weight-norm names to the old-style ones.

    torch >=2.1 stores weight norm as `parametrizations.weight.original0/1`,
    while Piper's VITS still uses the classic `weight_g` / `weight_v`. The
    tensors are the same (g = original0, v = original1); only the names moved.
    """
    marker = ".parametrizations.weight.original"
    if marker not in key:
        return key
    prefix, which = key.split(marker)
    return prefix + (".weight_g" if which == "0" else ".weight_v")


def _remap_flow_index(key):
    """Renumber flow layers from Coqui's packing to Piper's.

    Both build the same normalizing flow, but Coqui's ModuleList holds only the
    coupling layers (0,1,2,3) while Piper's also holds the parameterless Flip
    layers between them (0,2,4,6). The stochastic duration predictor is the
    same story with a leading ElementwiseAffine, giving 0,1,3,5,7.

    Without this every coupling layer would load into the wrong slot, and the
    export would produce confident-sounding noise rather than speech.
    """
    for prefix, positions in (
        ("flow.flows.", (0, 2, 4, 6)),
        ("dp.flows.", (0, 1, 3, 5, 7)),
        ("dp.post_flows.", (0, 1, 3, 5, 7)),
    ):
        if key.startswith(prefix):
            rest = key[len(prefix):]
            idx, _, tail = rest.partition(".")
            if idx.isdigit() and int(idx) < len(positions):
                return f"{prefix}{positions[int(idx)]}.{tail}"
    return key


# Coqui and Piper give ElementwiseAffine's two parameters different names.
# Same tensors, same shapes — only the attribute names differ.
_PARAM_RENAME = {".translation": ".m", ".log_scale": ".logs"}


def _remap_param_name(key):
    for src, dst in _PARAM_RENAME.items():
        if key.endswith(src):
            return key[: -len(src)] + dst
    return key


def remap_state_dict(coqui_sd):
    """Rename Coqui generator tensors to Piper's names, dropping the discriminator."""
    out, dropped = {}, 0
    for key, value in coqui_sd.items():
        if key.startswith("disc."):
            dropped += 1        # training-only discriminator
            continue
        for src, dst in _RENAME.items():
            if key.startswith(src):
                key = dst + key[len(src):]
                break
        key = _remap_flow_index(_denormalize_weight_norm(key))
        out[_remap_param_name(key)] = value
    return out, dropped


def infer_architecture(sd):
    """Read the architecture off the tensors rather than the config.

    Coqui writes `null` for most model_args and relies on its own defaults, so
    the checkpoint itself is the only reliable description of what was trained.
    """
    def shape(key):
        return tuple(sd[key].shape)

    n_vocab, hidden = shape("text_encoder.emb.weight")
    filter_channels = shape("text_encoder.encoder.ffn_layers.0.conv_1.weight")[0]
    n_layers = len({
        k.split(".")[3] for k in sd
        if k.startswith("text_encoder.encoder.attn_layers.")
    })
    n_heads = hidden // shape("text_encoder.encoder.attn_layers.0.emb_rel_k")[2]
    # proj emits mean and logvar concatenated, hence the halving.
    inter_channels = shape("text_encoder.proj.weight")[0] // 2
    spec_channels = shape("posterior_encoder.pre.weight")[1]
    upsample_initial = shape("waveform_decoder.conv_pre.weight")[0]

    ups_idx = sorted(
        {k.split(".")[2] for k in sd if k.startswith("waveform_decoder.ups.")},
        key=int,
    )
    upsample_kernel_sizes, upsample_rates = [], []
    for i in ups_idx:
        # Weight-norm stores the real kernel under parametrizations.
        w = sd.get(f"waveform_decoder.ups.{i}.parametrizations.weight.original1")
        if w is None:
            w = sd[f"waveform_decoder.ups.{i}.weight"]
        kernel = w.shape[2]
        upsample_kernel_sizes.append(kernel)
        # HiFiGAN builds each stage as ConvTranspose(kernel=2*rate, stride=rate).
        upsample_rates.append(kernel // 2)

    n_resblocks = len({
        k.split(".")[2] for k in sd if k.startswith("waveform_decoder.resblocks.")
    })
    kernels_per_stage = n_resblocks // len(ups_idx)
    resblock_kernel_sizes = []
    for j in range(kernels_per_stage):
        w = sd.get(f"waveform_decoder.resblocks.{j}.convs1.0.parametrizations.weight.original1")
        if w is None:
            w = sd[f"waveform_decoder.resblocks.{j}.convs1.0.weight"]
        resblock_kernel_sizes.append(w.shape[2])

    use_sdp = any(k.startswith("duration_predictor.flows.") for k in sd)

    return {
        "n_vocab": n_vocab,
        "spec_channels": spec_channels,
        "inter_channels": inter_channels,
        "hidden_channels": hidden,
        "filter_channels": filter_channels,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "upsample_initial_channel": upsample_initial,
        "upsample_rates": upsample_rates,
        "upsample_kernel_sizes": upsample_kernel_sizes,
        "resblock_kernel_sizes": resblock_kernel_sizes,
        "use_sdp": use_sdp,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Coqui best_model.pth")
    ap.add_argument("--config", required=True, help="Coqui config.json")
    ap.add_argument("--output-dir", required=True,
                    help="voice.onnx and voice.onnx.json are written here")
    ap.add_argument("--espeak-voice", default=None,
                    help="espeak voice code recorded in the JSON (metadata only; "
                         "a character model does not phonemize)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("use_phonemes"):
        print("This model uses phonemes; only character models convert cleanly.",
              file=sys.stderr)
        return 1

    vocab = build_vocab(cfg["characters"])
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    coqui_sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt

    arch = infer_architecture(coqui_sd)
    if arch["n_vocab"] != len(vocab):
        # The embedding is the ground truth. A mismatch means the vocab was
        # rebuilt wrongly, and every id would be off — silent gibberish audio.
        print(f"vocab size {len(vocab)} != embedding rows {arch['n_vocab']}; "
              "refusing to export a model whose ids would be misaligned.",
              file=sys.stderr)
        return 1
    print(f"vocab {len(vocab)} matches embedding ✓")
    print("architecture:", json.dumps(arch, indent=2))

    from piper.train.vits.models import SynthesizerTrn

    model = SynthesizerTrn(
        n_vocab=arch["n_vocab"],
        spec_channels=arch["spec_channels"],
        segment_size=32,           # training-only; unused at inference
        inter_channels=arch["inter_channels"],
        hidden_channels=arch["hidden_channels"],
        filter_channels=arch["filter_channels"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=arch["resblock_kernel_sizes"],
        resblock_dilation_sizes=[[1, 3, 5]] * len(arch["resblock_kernel_sizes"]),
        upsample_rates=arch["upsample_rates"],
        upsample_initial_channel=arch["upsample_initial_channel"],
        upsample_kernel_sizes=arch["upsample_kernel_sizes"],
        n_speakers=0,
        use_sdp=arch["use_sdp"],
    )

    piper_sd, dropped = remap_state_dict(coqui_sd)
    missing, unexpected = model.load_state_dict(piper_sd, strict=False)
    # enc_q is the posterior encoder: used only during training, absent at
    # inference, so it is expected to be unused here.
    real_missing = [k for k in missing if not k.startswith("enc_q.")]
    print(f"dropped {dropped} discriminator tensors")
    if real_missing:
        print(f"WARNING: {len(real_missing)} missing tensors, e.g. {real_missing[:5]}",
              file=sys.stderr)
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected tensors, e.g. {unexpected[:5]}",
              file=sys.stderr)
    if real_missing:
        print("Refusing to export with missing generator weights.", file=sys.stderr)
        return 1

    model.eval()
    with torch.no_grad():
        model.dec.remove_weight_norm()

    def infer_forward(text, text_lengths, scales, sid=None):
        return model.infer(
            text, text_lengths,
            noise_scale=scales[0], length_scale=scales[1], noise_scale_w=scales[2],
            sid=sid,
        )[0].unsqueeze(1)

    model.forward = infer_forward

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "voice.onnx"

    dummy_len = 50
    seq = torch.randint(low=0, high=arch["n_vocab"], size=(1, dummy_len),
                        dtype=torch.long)
    seq_lengths = torch.LongTensor([seq.size(1)])
    scales = torch.FloatTensor([0.667, 1.0, 0.8])

    export_kwargs = dict(
        opset_version=15,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time1", 2: "time2"},
        },
        do_constant_folding=True,
    )
    # torch 2.9+ defaults to the torch.export path, which cannot trace the
    # `assert (discriminant >= 0).all()` inside the spline transform the
    # stochastic duration predictor uses — it is a training-time sanity check
    # on a data-dependent value. The legacy TorchScript exporter traces
    # straight through it, and this is exactly the graph Piper's own exporter
    # produces, so ask for it explicitly when the option exists.
    if "dynamo" in torch.onnx.export.__doc__ or hasattr(torch.onnx, "dynamo_export"):
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        (seq, seq_lengths, scales, None),
        str(onnx_path),
        **export_kwargs,
    )
    print("wrote", onnx_path)

    audio = cfg.get("audio", {})

    # Reconcile the two tokenizers. Coqui (add_blank=True, use_eos_bos=False)
    # emits  [BLNK, c, BLNK, c, ..., BLNK]  with no sentence markers, while
    # Piper always emits  [BOS, PAD, c, PAD, ..., EOS].
    #
    # Pointing Piper's BOS/EOS/PAD at Coqui's blank id makes the two sequences
    # identical, so the model sees precisely the token pattern it was trained
    # on. Without this the ids are shifted and the audio is noise.
    id_map = {token: [i] for i, token in enumerate(vocab)}
    blank_id = len(vocab) - 1                      # <BLNK>, last by construction
    if cfg.get("add_blank", True):
        id_map["_"] = [blank_id]                   # Piper PAD  -> Coqui blank
        id_map["^"] = [blank_id]                   # Piper BOS  -> Coqui blank
        id_map["$"] = [blank_id]                   # Piper EOS  -> Coqui blank
    else:
        # No interleaving during training: neutralize Piper's padding instead.
        id_map["_"] = []
        id_map["^"] = []
        id_map["$"] = []

    voice_json = {
        "audio": {
            "sample_rate": audio.get("sample_rate", 22050),
            "quality": "medium",
        },
        "espeak": {"voice": args.espeak_voice or ""},
        # "text" tells Piper these ids are characters, not espeak phonemes, so
        # it skips phonemization and maps the raw text through phoneme_id_map.
        "phoneme_type": "text",
        "phoneme_id_map": id_map,
        "num_symbols": arch["n_vocab"],
        "num_speakers": 1,
        "speaker_id_map": {},
        "inference": {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w": 0.8,
        },
        "language": {"code": args.espeak_voice or ""},
    }
    json_path = out_dir / "voice.onnx.json"
    json_path.write_text(json.dumps(voice_json, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print("wrote", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
