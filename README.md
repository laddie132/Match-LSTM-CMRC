# Match-LSTM-CMRC(GM-Reader)

a version of [Match-LSTM+](https://github.com/laddie132/Match-LSTM) for CMRC dataset.

## Helps
the same as [Match-LSTM+](https://github.com/laddie132/Match-LSTM).

check the config file `config/global_config.yaml` before running.

## Results

### Test
https://hfl-rc.github.io/cmrc2018/leaderboard/

|Rank|Model|Test-EM|Test-F1|Test-Ave|
|---|---|---|---|---|
|7|GM-Reader(ensemble)|64.045|83.046|73.546|
|9|GM-Reader(single)|60.470|80.035|70.252|

### Dev
https://hfl-rc.github.io/cmrc2018/leaderboard_dev/

|Rank|Model|Dev-EM|Dev-F1|Dev-Ave|
|---|---|---|---|---|
|3|GM-Reader(ensemble)|63.900|83.540|73.720|
|7|GM-Reader(single)|60.750|80.480|70.615|

> GM-Reader is another name for Match-LSTM+.

## Links
- CMRC 2018[https://hfl-rc.github.io/cmrc2018/]
- Match-LSTM+[https://github.com/laddie132/Match-LSTM]
- Chinese-Word-Vectors[https://github.com/Embedding/Chinese-Word-Vectors]
