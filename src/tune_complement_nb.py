from src.tune_utils import run_grid_search


def main() -> None:
    results = run_grid_search(model_name="complement_nb", scoring="f1_macro")
    print("Complement Naive Bayes tuning complete.")
    print(results.head().to_string(index=False))


if __name__ == "__main__":
    main()
