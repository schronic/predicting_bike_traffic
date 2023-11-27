from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def predict_final_test(pipe):
    
    final_test = pd.read_parquet(Path("../data") / "test_final.parquet")
    final_test_pred = pipe.predict(final_test)
    submission = pd.DataFrame(final_test_pred, columns=['log_bike_count']).reset_index(names=['Id'])
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    submission.to_csv(f'../submissions/{timestamp}', index=False)
    
    
def sample_day_plot(X_test, y_pred, y_test, title, path):
    mask = (
        (X_test["counter_name"] == "Totem 73 boulevard de Sébastopol S-N")
        & (X_test["date"] > pd.to_datetime("2021/09/01"))
        & (X_test["date"] < pd.to_datetime("2021/09/02"))
    )

    df_viz = X_test.loc[mask].copy()
    df_viz["bike_count"] = np.exp(y_test[mask.values]) - 1
    df_viz["bike_count (predicted)"] = np.exp(y_pred[mask]) - 1
    
    fig, ax = plt.subplots(figsize=(12, 4))

    df_viz.plot(x="date", y="bike_count", ax=ax)
    df_viz.plot(x="date", y="bike_count (predicted)", ax=ax, ls="--")
    ax.set_title(f"Predictions for {title}")
    ax.set_ylabel("bike_count")
    plt.savefig(f'{path}/sample_day_plot_{title}.png')
    
def sample_week_plot(X_test, y_pred, y_test, title, path):
    mask = (
        (X_test["counter_name"] == "Totem 73 boulevard de Sébastopol S-N")
        & (X_test["date"] > pd.to_datetime("2021/09/01"))
        & (X_test["date"] < pd.to_datetime("2021/09/08"))
    )

    df_viz = X_test.loc[mask].copy()
    df_viz["bike_count"] = np.exp(y_test[mask.values]) - 1
    df_viz["bike_count (predicted)"] = np.exp(y_pred[mask]) - 1
    
    fig, ax = plt.subplots(figsize=(12, 4))

    df_viz.plot(x="date", y="bike_count", ax=ax)
    df_viz.plot(x="date", y="bike_count (predicted)", ax=ax, ls="--")
    ax.set_title(f"Predictions for {title}")
    ax.set_ylabel("bike_count")
    plt.savefig(f'{path}/sample_week_plot_{title}.png')
    
    
def residual_plot(y_pred, y_test, title, path):
    fig, ax = plt.subplots()

    plt.title(f"Residual plot for {title}")
    df_viz = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).sample(
        10000, random_state=0
    )

    df_viz.plot.scatter(x="y_true", y="y_pred", s=8, alpha=0.1, ax=ax)
    plt.savefig(f'{path}/residual_plot_{title}.png')
    plt.show()
    
    

def rmse_test_train_plot(params, title, path):
    test_values = ['split0_test_score', 'split1_test_score', 'split2_test_score',
       'split3_test_score', 'split4_test_score', 'split5_test_score',
       'split6_test_score', 'split7_test_score']
    train_values = ['split0_train_score',
       'split1_train_score', 'split2_train_score', 'split3_train_score',
       'split4_train_score', 'split5_train_score', 'split6_train_score',
       'split7_train_score']

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(range(1, len(train_values) + 1), params[train_values].values[0], 'b-', label='Train RMSE')
    ax.plot(range(1, len(train_values) + 1), params[test_values].values[0], 'r-', label='Test RMSE')
    
 
    graph_title = f"{title}: Train vs Test -RMSE/ Fold"
    plt.title(graph_title)
    plt.xlabel('Nr of Fold')
    plt.ylabel('-RMSE')

    plt.legend()
    plt.grid()
    
    plt.savefig(f'{path}/rmse_test_train_plot_{title}.png')
    plt.show()
    
    
def feature_importance(features, importances, title, path):
    # Figure Size
    fig, ax = plt.subplots(figsize =(16, 9))

    # Horizontal Bar Plot
    ax.barh(features, importances)

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)


    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)

    # Add x, y gridlines
    ax.grid(color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)

    # Show top values 
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5, 
                 str(round((i.get_width()), 2)),
                 fontsize = 10, fontweight ='bold',
                 color ='grey')

    title = f'{title}: Top 10 Feature Importances'
    ax.set_title(title,
                 loc ='left')
    
    plt.savefig(f'{path}/{title}.png')
    plt.show()