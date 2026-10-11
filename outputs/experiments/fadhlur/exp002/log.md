# fadhlur/exp002 tuned LightGBM

- seed: 42
- features: 44
- params: {'n_estimators': 200, 'learning_rate': 0.03, 'num_leaves': 31, 'min_child_samples': 100}
- OOF KS: 0.38528
- fold 0: 0.38934
- fold 1: 0.37722
- fold 2: 0.40226
- fold 3: 0.39202
- fold 4: 0.38074

## Tuning grid
- OOF 0.38528 {'n_estimators': 200, 'learning_rate': 0.03, 'num_leaves': 31, 'min_child_samples': 100}
- OOF 0.38493 {'n_estimators': 200, 'learning_rate': 0.03, 'num_leaves': 31, 'min_child_samples': 20}
- OOF 0.38395 {'n_estimators': 200, 'learning_rate': 0.03, 'num_leaves': 63, 'min_child_samples': 100}
- OOF 0.38341 {'n_estimators': 200, 'learning_rate': 0.03, 'num_leaves': 63, 'min_child_samples': 20}
- OOF 0.38220 {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 20}
- OOF 0.38213 {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 63, 'min_child_samples': 20}
- OOF 0.38204 {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 100}
- OOF 0.38182 {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 63, 'min_child_samples': 100}
- OOF 0.38107 {'n_estimators': 200, 'learning_rate': 0.03, 'num_leaves': 127, 'min_child_samples': 100}
- OOF 0.37935 {'n_estimators': 200, 'learning_rate': 0.03, 'num_leaves': 127, 'min_child_samples': 20}
- OOF 0.37381 {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 20}
- OOF 0.36987 {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 100}
