"""
generate_dataset.py
--------------------
Generates a synthetic but realistic student academic/skill dataset used to
train the Best-Fit Career Field Predictor.

Why synthetic data?
Real labeled "student -> best career field" datasets don't publicly exist
(no university releases this), so we simulate one using realistic
distributions per field. Each field has a distinct profile of academic
scores, skill ratings, and activity counts, with noise added so the model
has to actually learn patterns instead of memorizing rules.

Run:
    python generate_dataset.py
Output:
    students.csv  (in this folder)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

FIELDS = ["AI-ML", "Web Development", "Core CS", "Data Science", "Cybersecurity"]
N_PER_FIELD = 400  # 5 x 400 = 2000 rows total

# Each field has a "profile": mean values for every feature.
# Format: [dsa, oop, dbms, os, networks, math_stats,
#          coding_hours_week, projects_count, internships_count,
#          ml_skill, web_skill, security_skill, design_skill,
#          hackathons, github_repos, certifications, leadership_score,
#          communication_score, problem_solving_score, research_interest]
PROFILES = {
    "AI-ML": dict(
        dsa=78, oop=70, dbms=65, os=60, networks=55, math_stats=85,
        coding_hours_week=14, projects_count=4, internships_count=1,
        ml_skill=85, web_skill=40, security_skill=35, design_skill=30,
        hackathons=2, github_repos=6, certifications=3, leadership_score=55,
        communication_score=60, problem_solving_score=82, research_interest=80
    ),
    "Web Development": dict(
        dsa=60, oop=75, dbms=70, os=55, networks=50, math_stats=55,
        coding_hours_week=16, projects_count=6, internships_count=1,
        ml_skill=30, web_skill=88, security_skill=35, design_skill=70,
        hackathons=3, github_repos=8, certifications=2, leadership_score=60,
        communication_score=70, problem_solving_score=65, research_interest=35
    ),
    "Core CS": dict(
        dsa=88, oop=80, dbms=72, os=82, networks=70, math_stats=70,
        coding_hours_week=12, projects_count=3, internships_count=1,
        ml_skill=40, web_skill=35, security_skill=45, design_skill=25,
        hackathons=2, github_repos=5, certifications=2, leadership_score=50,
        communication_score=55, problem_solving_score=88, research_interest=55
    ),
    "Data Science": dict(
        dsa=70, oop=60, dbms=78, os=55, networks=45, math_stats=88,
        coding_hours_week=11, projects_count=4, internships_count=1,
        ml_skill=75, web_skill=30, security_skill=30, design_skill=35,
        hackathons=1, github_repos=5, certifications=3, leadership_score=55,
        communication_score=65, problem_solving_score=75, research_interest=70
    ),
    "Cybersecurity": dict(
        dsa=68, oop=65, dbms=60, os=85, networks=88, math_stats=60,
        coding_hours_week=13, projects_count=3, internships_count=1,
        ml_skill=25, web_skill=40, security_skill=90, design_skill=20,
        hackathons=4, github_repos=4, certifications=4, leadership_score=50,
        communication_score=55, problem_solving_score=78, research_interest=45
    ),
}

FEATURES = list(next(iter(PROFILES.values())).keys())


def sample_field(field_name, n):
    profile = PROFILES[field_name]
    rows = []
    for _ in range(n):
        row = {}
        for feat in FEATURES:
            mean = profile[feat]
            # noise scaled relative to typical magnitude of the feature
            noise_scale = max(mean * 0.30, 5)
            val = np.random.normal(mean, noise_scale)
            # clip score-like features to sane 0-100 range, counts to >=0
            if feat in ("coding_hours_week", "projects_count", "internships_count",
                        "hackathons", "github_repos", "certifications"):
                val = max(0, round(val))
            else:
                val = float(np.clip(val, 0, 100))
            row[feat] = val
        row["best_field"] = field_name
        rows.append(row)
    return rows


def main():
    all_rows = []
    for field in FIELDS:
        all_rows.extend(sample_field(field, N_PER_FIELD))

    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df.to_csv("students.csv", index=False)
    print(f"Saved students.csv with {len(df)} rows and {len(FEATURES)} features.")
    print(df["best_field"].value_counts())


if __name__ == "__main__":
    main()
