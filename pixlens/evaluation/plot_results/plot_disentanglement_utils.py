import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json


with open(
    r"C:\Users\lluis\AppData\Local\pixlens\pixlens\Cache\models--InstructPix2Pix\disentanglement\results.json",
    "r",
) as f:
    pix2pix = json.load(f)
with open(
    r"C:\Users\lluis\AppData\Local\pixlens\pixlens\Cache\models--ControlNet\disentanglement\results.json",
    "r",
) as f:
    controlnet = json.load(f)
with open(
    r"C:\Users\lluis\AppData\Local\pixlens\pixlens\Cache\models--LCM\disentanglement\results.json",
    "r",
) as f:
    lcm = json.load(f)

data = {}
data.update(pix2pix)
data.update(controlnet)
data.update(lcm)


def compare_avg_norms(data: dict) -> None:
    models = list(data.keys())
    attributes = ["texture", "color", "style", "pattern"]

    # Creating a DataFrame for the comparison
    rows = []

    for model in models:
        for attribute in attributes:
            avg_norm = data[model]["Avg_norm_per_attribute"][attribute]
            rows.append(
                {"Model": model, "Attribute": attribute, "Avg Norm": avg_norm},
            )
    
    df = pd.concat([pd.DataFrame([row]) for row in rows], ignore_index=True)
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Attribute", y="Avg Norm", hue="Model", data=df)
    plt.title("Intra-attribute scores")
    plt.ylabel("Average Norm")
    plt.xlabel("Attribute")
    plt.legend(title="Model")
    plt.show()


compare_avg_norms(data)
