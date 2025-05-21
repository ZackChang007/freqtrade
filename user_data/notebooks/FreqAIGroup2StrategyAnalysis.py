# 运行当前脚本时，临时把conda env切换为freqtrade env

# %% codecell
import sys

print(f"当前Python路径: {sys.executable}")
print(f"当前Python版本: {sys.version}")

# %%
from pathlib import Path
import os

# 显示Jupyter文件的当前路径，即ft_userdata repo下的路径，调用其他脚本时注意其路径和cwd的相对关系
print(Path.cwd())

# 设置路径为freqtrade repo的根目录

project_root = "C:\\Users\\Zack Chang\\Documents\\freqtrade"
i=0
try:
    os.chdir(project_root)
    assert Path('LICENSE').is_file()
except Exception as e:
    print(e)
    while i<4 and (not Path('LICENSE').is_file()):
        os.chdir(Path(Path.cwd(), '../'))
        i+=1
    project_root = Path.cwd()
print(Path.cwd())

# %%
from freqtrade.configuration import Configuration, TimeRange
# 使用已有的config.json文件，指的是freqtrade repo下的配置文件
import json

config = Configuration.from_files(["./user_data/config.json"])
# config
print(json.dumps(config['original_config'], indent=2))

# %%
# # Define some constants
# config["timeframe"] = "4h"
# # Name of the strategy class
config["strategy"] = "FreqAIGroup2Strategy"
config["freqaimodel"] = "XGBoostRegressor"
config["timerange"] = "20250101-20250501"
# Location of the data
data_location = config['datadir']
config["dataformat_ohlcv"] = "feather"
config["freqai"]["feature_parameters"]["buffer_train_data_candles"] = 0
# # Pair to analyze - Only use one pair here
pair = "ACE/USDT:USDT"

# %%
# Load data using values set above
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType

candles = load_pair_history(datadir=data_location,
                            timeframe=config["timeframe"],
                            pair=pair,
                            data_format = "feather",
                            candle_type=CandleType.FUTURES,
                            )

# Confirm success
print(f"Loaded {len(candles)} rows of data for {pair} from {data_location}")
candles.tail()

# %%
import nest_asyncio
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver
from freqtrade.data.dataprovider import DataProvider
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

nest_asyncio.apply()

exchange = ExchangeResolver.load_exchange(config, load_leverage_tiers=True)
strategy = StrategyResolver.load_strategy(config)
strategy.dp = DataProvider(config, exchange)
strategy.freqai = FreqaiModelResolver.load_freqaimodel(config)
strategy.freqai_info = config["freqai"]

freqai = strategy.freqai
freqai.live = True
freqai.dk = FreqaiDataKitchen(config)
freqai.dk.live = True

timerange = TimeRange.parse_timerange(config["timerange"])
freqai.dd.load_all_pair_histories(timerange, freqai.dk)

corr_dataframes, base_dataframes = freqai.dd.get_base_and_corr_dataframes(
    timerange, pair, freqai.dk
)

prediction_dataframe = base_dataframes[config["timeframe"]].copy()

unfiltered_df = freqai.dk.use_strategy_to_populate_indicators(
    strategy=strategy,
    corr_dataframes=corr_dataframes,
    base_dataframes=base_dataframes,
    pair=pair,
    prediction_dataframe=prediction_dataframe
)

freqai.dk.find_features(unfiltered_df)
freqai.dk.find_labels(unfiltered_df)

print(f"label list:{freqai.dk.label_list}")

# %%
# # Load strategy using values set above
features_filtered, labels_filtered = freqai.dk.filter_features(
    unfiltered_df,
    freqai.dk.training_features_list,
    freqai.dk.label_list,
    training_filter=True,
)

# # dk.normalize_data(), dk.make_train_test_datasets()在新版本中已经被丢弃
# # https://www.freqtrade.io/en/stable/strategy_migration/#freqai-new-data-pipeline
# data_dictionary = freqai.dk.make_train_test_datasets(features_filtered, labels_filtered)
# data_dictionary = freqai.dk.normalize_data(data_dictionary)
# # data_cleaning_train()在新版本中已经被丢弃: https://www.freqtrade.io/en/stable/strategy_migration/#freqai-new-data-pipeline
# freqai.data_cleaning_train(freqai.dk)

dd = freqai.dk.make_train_test_datasets(features_filtered, labels_filtered)
freqai.dk.feature_pipeline = freqai.define_data_pipeline(threads=freqai.dk.thread_count)
freqai.dk.label_pipeline = freqai.define_label_pipeline(threads=freqai.dk.thread_count)
(dd["train_features"],
dd["train_labels"],
dd["train_weights"]) = freqai.dk.feature_pipeline.fit_transform(dd["train_features"],
                                                                dd["train_labels"],
                                                                dd["train_weights"])

(dd["test_features"],
dd["test_labels"],
dd["test_weights"]) = freqai.dk.feature_pipeline.transform(dd["test_features"],
                                                           dd["test_labels"],
                                                           dd["test_weights"])

dd["train_labels"], _, _ = freqai.dk.label_pipeline.fit_transform(dd["train_labels"])
dd["test_labels"], _, _ = freqai.dk.label_pipeline.transform(dd["test_labels"])

# %%
X = dd["train_features"]
y = dd["train_labels"]
w = dd["train_weights"]
XX = dd["test_features"]
yy = dd["test_labels"]
ww = dd["test_weights"]

# %%
from catboost import CatBoostRegressor, Pool

train_pool = Pool(data=X, label=y, weight=w)
test_pool = Pool(data=XX, label=yy, weight=ww)
model = CatBoostRegressor(
    n_estimators=1000,
    early_stopping_rounds=100,
    allow_writing_files=False,
    verbose=False,
    random_seed=1,
    eval_metric="RMSE"
)
model.fit(train_pool, eval_set=test_pool)
model.get_best_score()["validation"]["RMSE"]

# %%
strategy.ft_bot_start()

# %%
# # Generate buy/sell signals using strategy
df = strategy.analyze_ticker(candles, {'pair': pair})
df.tail()

# %%
# Report results
print(f"Generated {df['enter_long'].sum()} entry signals")
data = df.set_index('date', drop=False)
data.tail()

# %%
from pprint import pprint

pprint(strategy.__dict__.keys())
print("\n")
pprint(strategy.__dict__['ft_buy_params'])

# %%
print(strategy.__source__)

# %%
vars(strategy)['stoploss']

# %%
import inspect

for name, member in inspect.getmembers(strategy, inspect.isfunction):
    print(f'Function {name}:')
    print(inspect.getsource(member))
    print('---')

# %%
print(inspect.getsource(strategy.__init__))

# %%
print(inspect.getsource(strategy.populate_indicators))

# %%
print(inspect.getsource(strategy.populate_entry_trend))

# %%
print(inspect.getsource(strategy.populate_exit_trend))

# %%
print(f"Methods defined directly in {type(strategy).__name__}:")
for name, member in inspect.getmembers(strategy, predicate=inspect.isfunction):
    print(f"Function {name}:")
    print(inspect.getsource(member))
    print("---")

# %%
print(f"\nAll methods (including inherited) in {type(strategy).__name__}:")
for name, member in inspect.getmembers(strategy, predicate=inspect.ismethod):
    print(f"Method {name}:")
    try:
        print(inspect.getsource(member))
    except TypeError:
        print(f"Unable to retrieve source for {name}")
    print("---")

# %%
# 获取__init__方法的参数规范
init_args = inspect.getfullargspec(strategy.__init__)
init_arg_names = init_args.args[1:]  # 去掉self参数

# 获取实例的所有属性值
instance_vars = vars(strategy)
pprint(instance_vars)
print("===" * 80)
print("\n")

# 根据参数名获取参数值
for arg_name in init_arg_names:
    if arg_name in instance_vars:
        pprint(f"{arg_name}: {instance_vars[arg_name]}")

# %%
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats

# if backtest_dir points to a directory, it'll automatically load the last backtest file.
# 如果有多个回测结果，默认加载最新的结果；也可指定某个回测结果
backtest_dir = config["user_data_dir"] / "backtest_results"
# backtest_dir can also point to a specific file
# backtest_dir = config["user_data_dir"] / "backtest_results/backtest-result-2020-07-01_20-04-22.json"

# %%

# You can get the full backtest statistics by using the following command.
# This contains all information used to generate the backtest result.
stats = load_backtest_stats(backtest_dir)

strategy = 'FreqAIGroup2Strategy'
# All statistics are available per strategy, so if `--strategy-list` was used during backtest, this will be reflected here as well.
# Example usages:
# Get pairlist used for this backtest
stats['strategy'][strategy]['pairlist']
# %%
stats['strategy'][strategy]['results_per_pair']
# %%
# Get market change (average change of all pairs from start to end of the backtest period)
pprint(stats['strategy'][strategy]['market_change'])
# %%
# Maximum drawdown ()
# pprint(stats['strategy'][strategy]['max_drawdown'])

# %%
# Maximum drawdown start and end
pprint(stats['strategy'][strategy]['drawdown_start'])
pprint(stats['strategy'][strategy]['drawdown_end'])
# %%
# Get strategy comparison (only relevant if multiple strategies were compared)
stats['strategy_comparison']
# %%
# Load backtested trades as dataframe
trades = load_backtest_data(backtest_dir)

# Show value-counts per pair
trades.groupby("pair")["exit_reason"].value_counts()
# %%
# Plotting equity line (starting with 0 on day 1 and adding daily profit for each backtested day)

from freqtrade.configuration import Configuration
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats
import plotly.express as px
import pandas as pd

# strategy = 'SampleStrategy'
# config = Configuration.from_files(["user_data/config.json"])
# backtest_dir = config["user_data_dir"] / "backtest_results"

stats = load_backtest_stats(backtest_dir)
strategy_stats = stats['strategy'][strategy]

dates = []
profits = []
for date_profit in strategy_stats['daily_profit']:
    dates.append(date_profit[0])
    profits.append(date_profit[1])

equity = 0
equity_daily = []
for daily_profit in profits:
    equity_daily.append(equity)
    equity += float(daily_profit)


df = pd.DataFrame({'dates': dates,'equity_daily': equity_daily})

fig = px.line(df, x="dates", y="equity_daily")
fig.show()
# %%
from freqtrade.data.btanalysis import load_trades_from_db

# Fetch trades from database
# /sqlite:///tradesv3.dry_run.sqlite
trades = load_trades_from_db("sqlite:///tradesv3.sqlite")

# Display results
trades.groupby("pair")["exit_reason"].value_counts()

# %%
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 在左侧axes上绘制profit_ratio的直方图:
trades['profit_ratio'].plot(kind='hist', ax=ax1, color='b', alpha=0.5)
ax1.set_xlabel('Profit Ratio')
ax1.set_ylabel('Frequency')
ax1.set_title('Profit Ratio Histogram')

# 在右侧axes上绘制profit_abs的直方图:
trades['profit_abs'].plot(kind='hist', ax=ax2, color='r', alpha=0.5)
ax2.set_xlabel('Profit Absolute')
ax2.set_ylabel('Frequency')
ax2.set_title('Profit Absolute Histogram')

# 调整子图间距并显示图表:
plt.subplots_adjust(wspace=0.5)
plt.show()

# %%
from freqtrade.data.btanalysis import analyze_trade_parallelism
# import matplotlib

# Load backtested trades as dataframe
trades = load_backtest_data(backtest_dir)
# # 加载sqlite中的实盘交易trades
# trades = load_trades_from_db("sqlite:///tradesv3.sqlite")

# Analyze the above
parallel_trades = analyze_trade_parallelism(trades, '5m')

# %matplotlib inline
parallel_trades.plot(backend="plotly")
# %%
from freqtrade.plot.plotting import  generate_candlestick_graph
# Limit graph period to keep plotly quick and reactive

# Filter trades to one pair
trades_red = trades.loc[trades['pair'] == pair]

data_red = data['2023-01-01':'2023-02-01']
# Generate candlestick graph
graph = generate_candlestick_graph(pair=pair,
                                   data=data_red,
                                   trades=trades_red,
                                   indicators1=['sma20', 'ema50', 'ema55'],
                                   indicators2=['rsi', 'macd', 'macdsignal', 'macdhist']
                                  )

# %%
# Show graph inline
graph.show()

# Render graph in a seperate window
# graph.show(renderer="browser")

# %%
import plotly.figure_factory as ff

hist_data = [trades.profit_ratio]
group_labels = ['profit_ratio']  # name of the dataset

fig = ff.create_distplot(hist_data, group_labels, bin_size=0.01)
fig.show()

# %%
path = r"C:\Users\Zack Chang\Documents\GitHub\ft_userdata\user_data\models\FreqAIGroup2Strategy"
# '/freqtrade/user_data/models/XGBoostRegressor/'
train_set = 'sub-train-ACE_1735689600'

# %%
symbol_timestamp = train_set.split("-")[-1].lower()
symbol_timestamp

# %%
import datetime

unix_dt = int(train_set.split('_')[-1])
unixToDatetime = datetime.datetime.fromtimestamp(unix_dt)
unixToDatetime

# %%
train_set_path = os.path.join(path, train_set)
os.listdir(train_set_path)

# %%
import pickle

def read_pkl(path, file):
    pkl = open(os.path.join(path, file), 'rb')
    data = pickle.load(pkl)
    pkl.close()
    return data

# %%
feature_pipeline = read_pkl(train_set_path, f'cb_{symbol_timestamp}_feature_pipeline.pkl')
print([d for d in dir(feature_pipeline) if "__" not in d])
pprint(feature_pipeline.__dict__)

# %%
feature_pipeline._convert_back_to_df

# %%
# 从steps可见，label的pipeline只做了scaler归一化处理

label_pipeline = read_pkl(train_set_path, f'cb_{symbol_timestamp}_label_pipeline.pkl')
print([d for d in dir(label_pipeline) if "__" not in d])
pprint(label_pipeline.__dict__)

# %%
from sklearn import set_config

set_config(display='diagram')
label_pipeline.steps[0]

# %%
import joblib

joblib_file = os.path.join(train_set_path, f'cb_{symbol_timestamp}_model.joblib')
model = joblib.load(joblib_file) 
# dir(model)
[d for d in dir(model) if "__" not in d]

# %%
model.__dict__

# %%
model.n_estimators, model.max_depth, model.max_leaves

# %%
import pandas as pd

importances = model.feature_importances_
names = model.feature_names_in_
feature_imp_df = pd.DataFrame({'feature':names, 'importance':importances})
feature_imp_df.sort_values('importance', ascending=False, inplace=True)
feature_imp_df

# %%
feature_imp_df.tail(30).plot.barh(x='feature', y='importance')
plt.xlabel("Feature Importance")
plt.title("Feature Importances")
plt.show()

# %%
# 训练集时间戳，series index从41开始是因为之前的数据dropna掉了

trained_dates = read_pkl(train_set_path, f'cb_{symbol_timestamp}_trained_dates_df.pkl')
print(trained_dates.head())
trained_dates.tail()

# %%
trained_data = read_pkl(train_set_path, f'cb_{symbol_timestamp}_trained_df.pkl').round(decimals=4)
# trained_data.round(decimals=4)

# %%
data_scaled = feature_pipeline['scaler'].transform(trained_data)
data_scaled

# %%
trained_data.iloc[:200, 10:20].plot(figsize=(12,8))

# %%
# historic_predictions_pkl = open(os.path.join(path, 'historic_predictions.pkl'), 'rb')
# historic_predictions_data = pickle.load(historic_predictions_pkl)
# keys = [k for k,v in historic_predictions_data.items()]
# for key in keys:
#     print(historic_predictions_data[key])
#     key_file_name = key.replace("/","-")
#     pd.DataFrame.from_dict(historic_predictions_data[key]).to_csv(f"{key_file_name}_historic_predictions.csv")

# %%
backtesting_predictions_folder = "backtesting_predictions"
backtesting_predictions_path = os.path.join(path, backtesting_predictions_folder)

prediction_files = [x for x in os.listdir(backtesting_predictions_path)]

# coin = 'eth'
# prediction_file = [x for x in prediction_files if (train_set.split('_')[-1] in x) & (coin in x)][0]
prediction_file = os.path.join(backtesting_predictions_path, f"cb_{symbol_timestamp}_prediction.feather")
prediction_file

# %%
import pyarrow.feather as feather

predict = feather.read_feather(prediction_file)

# %%
import json

metadata_file = (os.path.join(train_set_path, "cb_ace_1735689600_metadata.json"))
metadata = json.load(open(metadata_file))
metadata

# %%
X = feature_pipeline.fit_transform(trained_data)
len(X)

# %%
X_0 = X[0]
X_1 = X[1]
X_2 = X[2]

# %%
# # 缺失train set label数据 
# y = label_pipeline.fit_transform(metadata['target'])

# %%

