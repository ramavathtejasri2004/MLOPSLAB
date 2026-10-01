import os
import subprocess
import sys
from datetime import datetime

# ============================================================
# BOOKS DATASET - AUTOMATED MLOPS LIFECYCLE
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(
    REPORT_DIR,
    "automated_lifecycle_report.txt"
)


def run_step(step_name, script_path):

    print("\n" + "=" * 70)
    print(step_name)
    print("=" * 70)

    command = [
        sys.executable,
        script_path
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"❌ {step_name} FAILED")
        return False

    print(f"✅ {step_name} COMPLETED")
    return True


# ============================================================
# START
# ============================================================

start_time = datetime.now()

print("=" * 70)
print("BOOKS DATASET MLOPS AUTOMATED LIFECYCLE")
print("=" * 70)

print("Project:", PROJECT_ROOT)
print("Start time:", start_time)


# ============================================================
# PIPELINE STEPS
# ============================================================

steps = [

    (
        "STEP 1 - DATA VALIDATION",
        os.path.join("src", "validate_data.py")
    ),

    (
        "STEP 2 - REGISTERED MODEL VALIDATION",
        os.path.join("src", "validate_outputs.py")
    ),

]


# ============================================================
# RUN PIPELINE
# ============================================================

results = []

for step_name, script_path in steps:

    success = run_step(
        step_name,
        script_path
    )

    results.append(
        (step_name, success)
    )

    if not success:
        break


# ============================================================
# FINAL STATUS
# ============================================================

end_time = datetime.now()

all_success = (
    len(results) == len(steps)
    and all(success for _, success in results)
)

print("\n" + "=" * 70)
print("AUTOMATED LIFECYCLE SUMMARY")
print("=" * 70)

for step_name, success in results:

    status = "PASSED" if success else "FAILED"

    print(
        f"{step_name}: {status}"
    )


print("\nStart time:", start_time)
print("End time:", end_time)
print("Duration:", end_time - start_time)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "BOOKS DATASET MLOPS AUTOMATED LIFECYCLE REPORT\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        f"Start time: {start_time}\n"
    )

    file.write(
        f"End time: {end_time}\n"
    )

    file.write(
        f"Duration: {end_time - start_time}\n\n"
    )

    file.write("PIPELINE RESULTS\n")
    file.write("-" * 70 + "\n")

    for step_name, success in results:

        status = "PASSED" if success else "FAILED"

        file.write(
            f"{step_name}: {status}\n"
        )

    file.write("\n")

    if all_success:
        file.write(
            "FINAL RESULT: AUTOMATED LIFECYCLE PASSED\n"
        )
    else:
        file.write(
            "FINAL RESULT: AUTOMATED LIFECYCLE FAILED\n"
        )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

if all_success:

    print("🎉 AUTOMATED MLOPS LIFECYCLE PASSED")

else:

    print("❌ AUTOMATED MLOPS LIFECYCLE FAILED")

print("=" * 70)

print("\nReport saved to:")
print(REPORT_FILE)