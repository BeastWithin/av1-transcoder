# av1tx — batch AV1 transcoder

Analyzes videos with ffprobe and re-encodes them to AV1 using the GPU's hardware encoder
(`av1_vaapi`, AMD RX 9060 XT / RDNA4). Audio, subtitles, chapters and metadata are copied
unchanged. Python standard library only; needs `ffmpeg` and `ffprobe`.

## Usage

```bash
./av1tx.py --selftest                       # check the GPU AV1 encoder works
./av1tx.py -r --analyze ~/Videos            # report specs + plan, no encoding
./av1tx.py --sample 60 ~/Videos/movie.mkv   # encode 60 s to calibrate, writes movie.sample.av1.mkv
./av1tx.py -r ~/Videos -o ~/Videos_av1      # transcode a whole folder
./av1tx.py ~/Videos/movie.mkv --encoder svt # software SVT-AV1: smaller files, much slower
```

Useful options:

| option | meaning |
|---|---|
| `--ssim-target 0.98` | minimum SSIM vs. the original (1.0 = identical). Below this, quality is raised and the file re-encoded. |
| `--quality N` | starting quality. VAAPI: `global_quality` 0–255, lower = better. SVT: CRF, lower = better. Default is chosen from the source bitrate. |
| `--max-tries 4` | encode attempts per file |
| `--no-verify` | skip the SSIM check (faster, no quality guard) |
| `--delete-original` | delete the source after a verified, smaller AV1 file was written |
| `--overwrite` | replace an existing `*.av1.mkv` |

## Safety rules

- Outputs go to `<name>.av1.mkv` via a `.partial` temp file; they are renamed only after verification.
- A file is only accepted if the AV1 stream is valid, audio track count and duration match, the
  SSIM target is met, **and the output is smaller than the source**. Otherwise the original is kept.
- Originals are never deleted unless you pass `--delete-original`.

## Notes

- Sources that are not 8/10-bit 4:2:0 (e.g. 4:2:2, 12-bit) are skipped; the hardware AV1 encoder cannot take them.
- Interlaced sources are deinterlaced (yadif) before encoding.
- Savings depend on the source. Old, heavily compressed sources (MPEG-2, XviD, H.264 at low
  bitrate) usually shrink the most. Already efficient sources (HEVC/AV1 at good bitrate) may
  not shrink, and are then kept as they are.
- `-qp` is ignored by Mesa's `av1_vaapi`; the tool uses `-global_quality` with CQP instead.
