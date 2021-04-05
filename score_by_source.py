from collections_utilis import * 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
if __name__ == '__main__':

    df_score_ledroit = get_df_score_ledroit()
    df_score_lesoleil = get_df_score_lesoleil()
    df_score_lenouvelliste = get_df_score_lenouvelliste()
    df_score_lequotidien = get_df_score_lequotidien()
    df_score_latribune = get_df_score_latribune()
    df_score_lavoixdelest = get_df_score_lavoixdelest()

    total = [df_score_ledroit, df_score_lesoleil, df_score_lenouvelliste, df_score_lequotidien, df_score_latribune, df_score_lavoixdelest]

    df_score_total = pd.concat(total)


    df_score_total_horizontal_stacked_bar =df_score_total[["source", "point_view", "point_view5", "point_view10", "point_view30", "point_view60", "score"]]
    df_score_total_horizontal_stacked_bar_grouped_by_sum = df_score_total_horizontal_stacked_bar.groupby("source").sum().reset_index()
    sources = list(df_score_total_horizontal_stacked_bar_grouped_by_sum["source"])



    sns.set_theme(style="ticks")
    f, ax = plt.subplots(figsize=(7, 6))
    #ax = sns.stripplot(x="score", y="source", data=df_score_total, jitter =True)
    ax = sns.boxplot(x="score", y="source" , data=df_score_total, width=.6, palette="Set1", showfliers=False, orient="h")
    #sns.swarmplot(x="score", y="source", data=df_score_total, size=4, color=".3", linewidth=0)
    #plt.title("Scores selon les sources", fontsize=20)
    plt.ylabel("Sources", fontsize=18)
    plt.xlabel("Scores", fontsize=18)
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    ax.set_yticklabels(sources, rotation=0, fontsize=16)
    sns.despine(trim=True, left=True)

    plt.show()

    df_score_total_horizontal_stacked_bar =df_score_total[["source", "point_view", "point_view5", "point_view10", "point_view30", "point_view60", "score"]]
    df_score_total_horizontal_stacked_bar_grouped_by_sum = df_score_total_horizontal_stacked_bar.groupby("source").sum().reset_index()
    sources = list(df_score_total_horizontal_stacked_bar_grouped_by_sum["source"])
    df_score_total_horizontal_stacked_bar_grouped_by_sum = df_score_total_horizontal_stacked_bar_grouped_by_sum[["point_view","point_view5", "point_view10", "point_view30", "point_view60"]].div(df_score_total_horizontal_stacked_bar_grouped_by_sum.score, axis=0)
    df_score_total_horizontal_stacked_bar_grouped_by_sum["source"] = sources
    df_score_total_horizontal_stacked_bar_grouped_by_sum = df_score_total_horizontal_stacked_bar_grouped_by_sum.set_index("source")

    ax = df_score_total_horizontal_stacked_bar_grouped_by_sum.plot(kind= "barh", stacked=True)


    for rect in ax.patches:
    # Find where everything is located
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()

        # The height of the bar is the data value and can be used as the label
        label_text = f'{width * 100 :.2f}%'  # f'{width:.2f}' to format decimal values

        # ax.text(x, y, text)
        label_x = x + width / 2
        label_y = y + height / 2

        # only plot labels greater than given width
        if width > 0:
            ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=15)

    # move the legend
    ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', borderaxespad=0.,  fontsize=15)

    plt.xlabel("pourcentage", fontsize=20)
    ax.set_yticklabels(sources, rotation=0, fontsize=16)
    ax.set(ylabel="")
    #plt.title("Distribution des points selon les views par journal", fontsize=20)
    plt.show()
