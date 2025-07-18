
import matplotlib.pyplot as plt

def plot_residual_diagnostics(df, label_col="lgcost", prediction='prediction'):
    df_pd = df.select(prediction, label_col).toPandas()
    df_pd["residual"] = df_pd[label_col] - df_pd[prediction]

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))

    # Plot 1: Residuals vs Predictions
    axs[0].scatter(df_pd[prediction], df_pd["residual"], alpha=0.5)
    axs[0].axhline(0, color="red", linestyle="--")
    axs[0].set_xlabel("Predicted Values")
    axs[0].set_ylabel("Residuals")
    axs[0].set_title("Residuals vs Predicted Values")

    # Plot 2: Residuals Histogram
    axs[1].hist(df_pd["residual"], bins=50, edgecolor="k")
    axs[1].set_xlabel("Residual")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Distribution of Residuals")

    plt.tight_layout()
    plt.show()