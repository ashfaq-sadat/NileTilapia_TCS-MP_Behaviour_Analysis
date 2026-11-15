# Behaviour of Nile tilapia larvae exposed to triclosan and polyethylene microplastics

Video recordings, tracking code, statistical analysis and output files for the larval
behaviour assay of a toxicity study on Nile tilapia (*Oreochromis niloticus*) larvae exposed
to triclosan (TCS), polyethylene microplastics (PE-MPs) and their combination.

This repository is the supplementary material for the behaviour part of the study. It holds
everything needed to go from the raw videos to the tables and figures: six recordings, two
Python scripts, and the files those scripts produced.

## The assay in brief

Ten-day-old larvae were recorded in 6-well plates, one larva per well and one plate per
treatment. Each plate holds two larvae from each of three replicate tanks, so every
treatment has six larvae from three tanks (36 larvae and 18 tanks in total).

| Treatment | Video | What it is |
|---|---|---|
| Control | `Control.mp4` | Untreated larvae |
| DMSO | `DMSO.mp4` | Solvent control for TCS |
| Tween80 | `Tween80.mp4` | Control for the surfactant used to suspend the PE-MPs |
| MP | `MP.mp4` | PE-MPs, suspended with Tween 80 |
| TCS | `TCS.mp4` | Triclosan, dissolved in DMSO |
| MP+TCS | `MP+TCS.mp4` | Combined exposure, containing both DMSO and Tween 80 |

Wells are numbered left to right, top row first (W1 to W3), then the bottom row (W4 to W6).
Wells W1 and W4 hold larvae from replicate tank R1, W2 and W5 from R2, and W3 and W6 from R3.

Exposure concentrations, particle characteristics, husbandry and recording conditions are
given in the article.

## Repository contents

```
videos/               six recordings, one per treatment (MP4, H.264, 1920 x 1080, 25 fps, 3.0 to 3.4 min)
larval_tracker.py     step 1: finds the wells, tracks each larva, computes the endpoints
larval_analysis.py    step 2: mixed models, contrasts, tables and figures
results/              output of step 1
analysis/             output of step 2
requirements.txt      package versions used
CITATION.cff          how to cite the repository
LICENSE               MIT licence for the code
LICENSE-DATA          CC BY 4.0 licence for the videos, results, tables and figures
```

## Running the analysis

The scripts were written and run with Python 3.13 on Windows. They use relative paths, so
run them from the repository folder.

```
python -m venv .venv
.venv\Scripts\activate            (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python larval_tracker.py          reads videos/, writes results/
python larval_analysis.py         reads results/, writes analysis/
```

The tracker takes the treatment name from the video file name, so `TCS.mp4` becomes treatment
`TCS`. All settings sit in a block at the top of each script.

To redo only the statistics and figures, skip the tracker and run `larval_analysis.py` on the
CSV files already in `results/`.

## How the tracker works

`larval_tracker.py` handles each video on its own, in these steps.

1. **Background.** A median image is built from 100 frames spread evenly through the video.
   Because the larva moves, the median contains the plate without the animal.
2. **Well detection.** The plate is separated from the surroundings by Otsu thresholding. The
   six largest bright regions inside it are taken as the wells and ordered as a 2 x 3 grid. In
   each well the ring marking the well floor is located with a Hough circle transform, run
   twice: the second pass limits the radius to within 5% of the median of the first, so all
   six wells get a ring of consistent size.
3. **Detection.** In every frame the grey image is subtracted from the background. Pixels more
   than 25 grey levels darker than the background, after a 3 x 3 morphological opening, are
   grouped into blobs, and blobs under 15 pixels are discarded. Only the well floor is searched
   (1.02 x the floor radius), which keeps reflections in the well wall out.
4. **Linking.** The blob nearest the previous position is taken as the larva. A jump of more
   than 8 mm in one frame is rejected as a tracking error. If the larva is lost for more than
   10 frames, tracking restarts from the largest blob. Gaps of up to 10 frames are filled by
   linear interpolation.
5. **Scale.** Pixels are converted to millimetres from the detected floor ring, assuming a
   floor diameter of 34.8 mm (see *Calibration* below).
6. **Endpoints.** The first 2 s of each video are left out because the camera shakes when
   recording starts. Frame-to-frame steps shorter than 0.15 mm are treated as noise and set to
   zero.

### Endpoints

| Column | Definition |
|---|---|
| `total_distance_mm` | Sum of all steps over the analysed period |
| `mean_speed_mm_s` | Total distance divided by analysed time |
| `mean_speed_when_active_mm_s` | Mean speed over active frames only |
| `max_speed_mm_s` | Highest speed after a 3-frame rolling median |
| `pct_time_active` | Share of frames with speed above 2 mm s⁻¹ |
| `n_bouts` | Number of runs of at least 2 consecutive active frames |
| `pct_time_outer_zone` | Share of time spent outside the centre zone (thigmotaxis) |

The centre zone is a circle with 0.707 of the floor radius, which divides the well floor into
a centre and an outer ring of equal area. A larva moving at random would therefore spend
about half its time in each.

### Files in `results/`

- `larva_summary.csv` has one row per larva (36 rows): video, treatment, replicate, well, the
  seven endpoints, and four quality columns. `analysed_s` is the analysed duration,
  `detection_pct` the share of frames in which the larva was found, `px_per_mm` the scale
  used, and `calibration` whether that scale was measured or estimated.
- `larva_per_minute.csv` has the same endpoints in 60 s bins (three bins per larva). A final
  bin shorter than 30 s is dropped.
- `<treatment>_wells.jpg` shows the detected tracking area (red), the floor ring and the centre
  zone (blue) on the background image, with well and replicate labels.
- `<treatment>_tracks.jpg` adds the trajectory of each larva.

The two image types are there so that tracking can be checked by eye. In this data set the
larva was detected in at least 99.9% of frames in every well, and 178 to 205 s were analysed
per larva.

## Statistical analysis

`larval_analysis.py` fits, for each endpoint, a linear mixed model with treatment as a fixed
effect and replicate tank as a random intercept:

```
endpoint ~ treatment + (1 | tank)
```

The two larvae from one tank share the random intercept, which makes the tank, not the larva,
the experimental unit. Tank R1 of one treatment and tank R1 of another are different tanks
and are coded as such.

- **Overall treatment effect.** Likelihood-ratio test of the model with treatment against the
  intercept-only model, both fitted by maximum likelihood.
- **Planned contrasts.** Seven comparisons are estimated from the REML fit and tested with
  Wald z tests. P values are Holm-adjusted within each endpoint.
- **Diagnostics.** The tank and residual variances, the intraclass correlation and a
  Shapiro-Wilk test of the residuals are reported for every endpoint.
- **Fallback.** When the tank variance is estimated at zero and the mixed model does not
  converge, an ordinary linear model is fitted and the `model` column says so. This did not
  happen for any endpoint in this data set.

The planned contrasts compare each exposure with the vehicle it was delivered in:

| Contrast | Question |
|---|---|
| DMSO vs Control | Does the solvent alone have an effect? |
| Tween80 vs Control | Does the surfactant alone have an effect? |
| TCS vs DMSO | Effect of TCS beyond its solvent |
| MP vs Tween80 | Effect of PE-MPs beyond the surfactant |
| MP+TCS vs TCS | Does adding PE-MPs change the TCS response? |
| MP+TCS vs MP | Does adding TCS change the PE-MP response? |
| MP+TCS vs Control | Combined exposure against untreated larvae |

Setting `COMPARISONS = "tukey"` at the top of the script replaces the planned contrasts with
all pairwise comparisons, Tukey-adjusted, and labels the figures with compact letters instead
of brackets.

### Files in `analysis/`

| File | Content |
|---|---|
| `Table1_descriptive.csv` | Mean, SD, SE, median, minimum, maximum and n per treatment and endpoint |
| `Table2_mixed_model_tests.csv` | Treatment effect per endpoint, variance components, ICC, residual normality |
| `Table3_planned_contrasts.csv` | Difference, SE, 95% CI, percentage change, z, raw and Holm-adjusted p for each contrast |
| `Table3_pairwise_tukey.csv` | All pairwise comparisons, from a run with `COMPARISONS = "tukey"` |
| `Tables.xlsx` | Tables 1 and 2 and the planned contrasts in one workbook |
| `Fig_box_<endpoint>` | Box plot with every larva shown as a point |
| `Fig_bar_<endpoint>` | Mean ± SE bar chart with every larva shown as a point |
| `Fig_panel_box`, `Fig_panel_bar` | Six main endpoints in one full-width figure, panels A to F |
| `Fig_time_course` | Distance, activity and outer-zone time per minute |

Every figure is saved three times: PNG and LZW-compressed TIFF at 600 dpi, and vector PDF.
Brackets with asterisks mark significant planned contrasts (\* p < 0.05, \*\* p < 0.01,
\*\*\* p < 0.001, Holm-adjusted).

## Calibration

The scale is estimated in each video
from the detected well-floor ring and a nominal floor diameter of 34.8 mm, giving 6.93 to
7.32 pixels per mm across the six videos. The `calibration` column in `larva_summary.csv`
reads `estimated` for this reason.

Distances and speeds scale directly with this value. The percentage endpoints depend on it
only through the 2 mm s⁻¹ activity threshold and the 0.15 mm noise threshold. To use a
measured scale instead, set `PX_PER_MM` at the top of `larval_tracker.py`; the note at the end
of that script explains how to measure it in ImageJ/Fiji.

## Using the tracker on other recordings

The tracker expects a top-down view of one 6-well plate, fully in frame, with dark larvae on a
bright background and one animal per well. Put the videos in `videos/`, name each file after
its treatment, and adjust `REPLICATE_OF_WELL` if the wells map to replicates differently.
Check the `_wells.jpg` and `_tracks.jpg` images before trusting the numbers.

## Citation

If you use the code or the data, please cite this repository (see `CITATION.cff`, or the
"Cite this repository" button on GitHub). The reference to the accompanying article and its
DOI will be added here when the article is published.

## Licence

The code is released under the MIT License (`LICENSE`). The videos, tracking results, tables
and figures are released under CC BY 4.0 (`LICENSE-DATA`).

## Contact

Md. Ashfaq Sadat, Department of Fisheries Management, Bangladesh Agricultural University,
Mymensingh, Bangladesh. Questions and problems can be raised through the Issues tab of this
repository.
