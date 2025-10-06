import json
import pandas as pd
import glob


def boost_grade(grade: str) -> str:
    if grade == "poor":
        return "fair"
    elif grade == "fair":
        return "good"
    else:
        return grade


def main():
    llm_eval_files = glob.glob("/Users/antonypapadimitriou/PycharmProjects/deepfin-benchmarking/deepfin_benchmarking/output/evaluation/evaluation_v2/oai/*.json")
    morningstar_eval_files = glob.glob("/Users/antonypapadimitriou/PycharmProjects/deepfin-benchmarking/deepfin_benchmarking/output/evaluation/evaluation_v2/morningstar/*.json")

    llm_eval_dict = {
        "assumptions": pd.DataFrame(),
        "coherence": pd.DataFrame(),
        "comprehensiveness": pd.DataFrame(),
        "depth": pd.DataFrame(),
        "originality": pd.DataFrame(),
    }

    morningstar_eval_dict = {
        "assumptions": pd.DataFrame(),
        "coherence": pd.DataFrame(),
        "comprehensiveness": pd.DataFrame(),
        "depth": pd.DataFrame(),
        "originality": pd.DataFrame(),
    }

    for file in llm_eval_files:
        split_file_name = file.split("/")[-1].split("_")
        ticker = split_file_name[0]
        metric = split_file_name[2]
        with open(file, "r") as f:
            eval_dict: dict = json.load(f)
        grade = eval_dict["grade"].strip().lower()
        eval_df = pd.DataFrame([{"ticker": ticker, "metric": metric, "grade": grade}])
        llm_eval_dict[metric] = pd.concat([llm_eval_dict[metric], eval_df])

    for file in morningstar_eval_files:
        split_file_name = file.split("/")[-1].split("_")
        ticker = split_file_name[0]
        metric = split_file_name[3]
        with open(file, "r") as f:
            eval_dict: dict = json.load(f)
        grade = eval_dict["grade"].strip().lower()
        eval_df = pd.DataFrame([{"ticker": ticker, "metric": metric, "grade": grade}])
        morningstar_eval_dict[metric] = pd.concat([morningstar_eval_dict[metric], eval_df])

    for metric in ["assumptions", "coherence", "comprehensiveness", "depth", "originality"]:
        llm_results = llm_eval_dict[metric]
        morningstar_results = morningstar_eval_dict[metric]
        morningstar_results["grade"] = morningstar_results["grade"].map(boost_grade)

        print(f"LLM {metric}:")
        print(llm_results["grade"].value_counts() / len(llm_results["grade"]))
        print("\n\n")

        print(f"Morningstar {metric}:")
        print(morningstar_results["grade"].value_counts() / len(morningstar_results["grade"]))
        print("\n\n")


if __name__ == "__main__":
    main()
