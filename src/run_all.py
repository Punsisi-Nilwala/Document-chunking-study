"""One command to reproduce everything:  python -m src.run_all"""
from . import error_analysis, extract, make_figures, make_report, run_experiments, validate_questions
from .run_experiments import RES


def main():
    extract.load_pages(rebuild=True)
    bad = validate_questions.validate(verbose=False)
    assert not bad, f"{len(bad)} questions have evidence that is not in the handbook"
    run_experiments.main()
    make_figures.main()
    for size in (200, 500, 1000):
        df = error_analysis.analyse(size, 0.25)
        df.to_csv(RES / f"error_analysis_bm25_stem_{size}w_ov25.csv", index=False)
        print(f"error analysis @{size} words: {len(df)} wrong top-1 predictions")
    make_report.main()


if __name__ == "__main__":
    main()
