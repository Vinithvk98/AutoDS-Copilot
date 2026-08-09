"""The knowledge base, a curated corpus of data science guidance.

Each entry is a short, self-contained chunk with a title used as its citation.
These are retrieved to ground the assistant's insights, its recommendations, and
the Ask panel, so the advice is anchored to real practice rather than generated.
"""

# Each doc: id, title, text, tags (help retrieval and topical grouping).
KNOWLEDGE_BASE = [
    # --- data preparation ---
    {"id": "missing-data", "title": "Handling missing data",
     "tags": "missing nulls imputation median mode",
     "text": "Impute numeric columns with the median, which is robust to outliers, "
             "and categoricals with the most frequent value. Columns missing more than "
             "half their values are usually better dropped. Fit any imputer on the "
             "training split only, never the whole dataset, to avoid leakage."},
    {"id": "missing-mechanism", "title": "Why data is missing matters",
     "tags": "missing mechanism mcar mnar bias",
     "text": "Missing values are not always random. If a value is missing for a reason "
             "related to the outcome, imputing it can hide signal or add bias. It is "
             "worth adding a simple was-missing flag column so the model can use the "
             "fact that a value was absent."},
    {"id": "outliers", "title": "Dealing with outliers",
     "tags": "outliers iqr robust winsorize",
     "text": "Outliers are values far from the rest, often past 1.5 times the "
             "interquartile range. They can distort means and linear models. Options "
             "are to cap them, transform the column, or use tree based models that are "
             "not sensitive to them. Investigate before removing, some outliers are the "
             "most interesting rows."},
    {"id": "duplicates", "title": "Duplicate rows and columns",
     "tags": "duplicates redundant identical",
     "text": "Exact duplicate rows inflate the apparent size of the data and can leak "
             "between train and test. Remove them. Duplicate columns are redundant and "
             "should be reduced to one."},
    {"id": "encoding", "title": "Encoding categorical variables",
     "tags": "categorical onehot encoding cardinality",
     "text": "One hot encoding turns categories into binary columns and suits low "
             "cardinality features. For high cardinality, group rare levels, or use "
             "target or frequency encoding with care to avoid leakage. Tree models "
             "handle encoded categories well."},
    {"id": "scaling", "title": "Feature scaling",
     "tags": "scaling standardize normalize distance",
     "text": "Distance and gradient based models like KNN, SVM, and neural networks "
             "need features on a similar scale, so standardise them. Tree based models "
             "do not need scaling. Always fit the scaler on the training data only."},
    {"id": "feature-engineering", "title": "Feature engineering",
     "tags": "features engineering interactions dates",
     "text": "Better features usually beat fancier models. Extract parts of dates, "
             "build ratios and interactions, and aggregate related rows. Domain "
             "knowledge is the best source of features. Keep new features honest, they "
             "must be knowable at prediction time."},
    {"id": "types", "title": "Getting column types right",
     "tags": "types parsing dates numeric text",
     "text": "A date stored as text or a number stuck as a string will be skipped or "
             "mangled by most tools. Fix representation first, parse dates, convert "
             "numeric looking text, and trim whitespace, before analysis."},

    # --- leakage and validation ---
    {"id": "leakage", "title": "Avoiding data leakage",
     "tags": "leakage pipeline validation target",
     "text": "Leakage is when information the model would not have at prediction time "
             "sneaks into training, and it produces scores that collapse in production. "
             "Wrap all preprocessing in a pipeline fit only on training data, and drop "
             "features derived from or knowable only after the target."},
    {"id": "leakage-signals", "title": "Spotting leakage",
     "tags": "leakage perfect correlation suspicious",
     "text": "A feature that predicts the target almost perfectly is usually leakage, "
             "not luck. Warning signs are a single feature with very high correlation to "
             "a numeric target, or a category where every value maps to one outcome. A "
             "model that scores near perfect deserves suspicion."},
    {"id": "cross-validation", "title": "Cross-validation",
     "tags": "cross validation kfold reliable estimate",
     "text": "A single train and test split is noisy. K fold cross validation trains on "
             "several splits and averages, giving a more reliable estimate of how the "
             "model generalises, plus a spread that shows how stable it is. Use "
             "stratified folds for classification."},
    {"id": "train-test-split", "title": "Holding data back",
     "tags": "split holdout test generalisation",
     "text": "Always keep a portion of data unseen during training and report the score "
             "on it. A model evaluated on data it trained on will look far better than "
             "it really is."},
    {"id": "overfitting", "title": "Detecting overfitting",
     "tags": "overfitting regularisation gap variance",
     "text": "A large gap between training and validation scores means the model "
             "memorised the training data. Mitigate with regularisation, simpler models, "
             "more data, or cross validation. Perfect scores on small or synthetic data "
             "are a red flag."},

    # --- metrics ---
    {"id": "classification-metrics", "title": "Classification metrics",
     "tags": "accuracy f1 precision recall confusion",
     "text": "Accuracy is fine for balanced classes. For imbalanced problems prefer "
             "weighted F1, precision, and recall. Precision asks how many predicted "
             "positives were right, recall asks how many real positives were caught. A "
             "confusion matrix shows which classes are confused."},
    {"id": "roc-auc", "title": "ROC AUC",
     "tags": "roc auc ranking threshold probability",
     "text": "ROC AUC measures how well a model ranks a random positive above a random "
             "negative, independent of the threshold. Above 0.8 is strong, around 0.5 is "
             "no better than chance. It is useful when you care about ranking or will "
             "tune the decision threshold."},
    {"id": "class-imbalance", "title": "Class imbalance",
     "tags": "imbalance imbalanced rare class weight resampling smote minority",
     "text": "When one class dominates, accuracy is misleading because always guessing "
             "the majority looks good. Use class weighting, resampling such as SMOTE, or "
             "threshold tuning, and judge with F1, precision recall, and ROC AUC rather "
             "than accuracy."},
    {"id": "regression-metrics", "title": "Regression metrics",
     "tags": "r2 rmse mae error variance",
     "text": "R squared is the share of variance the model explains. RMSE and MAE are "
             "errors in the target's own units, and RMSE punishes big misses more. A "
             "negative R squared means the model does worse than guessing the average."},
    {"id": "clustering-metrics", "title": "Judging clusters",
     "tags": "silhouette clustering separation k",
     "text": "Without labels, the silhouette score measures how tight and separated the "
             "clusters are, from minus one to one, where higher is better. Above 0.5 is "
             "well separated. Also sanity check by describing what each cluster has in "
             "common."},

    # --- model families ---
    {"id": "model-linear", "title": "Linear and logistic models",
     "tags": "linear logistic baseline interpretable",
     "text": "Linear and logistic regression are fast, interpretable baselines. They "
             "work well when the relationship is roughly linear and features are scaled. "
             "Always start here to set a bar the fancier models must beat."},
    {"id": "model-trees", "title": "Trees and forests",
     "tags": "decision tree random forest ensemble",
     "text": "A single decision tree gives readable rules but overfits. Random forests "
             "and extra trees average many trees for a strong, low effort default that "
             "handles mixed types and non linear patterns without scaling."},
    {"id": "model-boosting", "title": "Gradient boosting",
     "tags": "gradient boosting xgboost lightgbm tabular",
     "text": "Gradient boosting, including XGBoost and LightGBM, builds trees "
             "sequentially to fix earlier mistakes and usually tops leaderboards on "
             "tabular data. It rewards a little tuning of the learning rate and number "
             "of trees, and can overfit if pushed too hard."},
    {"id": "model-svm", "title": "Support vector machines",
     "tags": "svm kernel margin scaled",
     "text": "SVMs find a boundary with the widest margin and shine on clean, scaled, "
             "medium sized data. They are slower on large datasets and need scaling and "
             "some tuning of the C parameter."},
    {"id": "model-knn", "title": "K nearest neighbours",
     "tags": "knn instance distance local",
     "text": "KNN predicts from the closest examples, so it captures local structure "
             "with no training step. It needs scaled features and slows down on large "
             "data, and it struggles when there are many features."},
    {"id": "model-nb", "title": "Naive Bayes",
     "tags": "naive bayes probabilistic text fast",
     "text": "Naive Bayes is a very fast probabilistic baseline that assumes features "
             "are independent. It is surprisingly good on text and high dimensional "
             "sparse data, and a fine sanity check even when the assumption does not "
             "hold."},
    {"id": "model-mlp", "title": "Neural networks for tables",
     "tags": "neural network mlp tabular",
     "text": "A small neural network can capture non linear patterns but usually needs "
             "more data and tuning than tree ensembles, which tend to win on typical "
             "tabular problems. Scale the inputs and watch for overfitting."},

    # --- model selection heuristics ---
    {"id": "select-tabular", "title": "Choosing a model for tabular data",
     "tags": "model selection tabular recommendation",
     "text": "For everyday tabular data, start with a linear baseline for a reference, "
             "then try random forest and gradient boosting, which usually win. Compare "
             "them with cross validation before committing rather than trusting one "
             "split."},
    {"id": "select-small-data", "title": "Small datasets",
     "tags": "small data simple regularise",
     "text": "With few rows, prefer simpler, regularised models and lean on cross "
             "validation, because complex models overfit small data and their scores are "
             "unreliable. More data usually helps more than a fancier model."},
    {"id": "select-highdim", "title": "Many features, few rows",
     "tags": "high dimensional regularisation selection",
     "text": "When there are many columns relative to rows, use regularised linear "
             "models such as Lasso or Ridge, or do feature selection first. This setting "
             "invites overfitting and spurious correlations."},
    {"id": "select-interpretable", "title": "When you need to explain the model",
     "tags": "interpretable explanation stakeholder",
     "text": "If a person has to trust or act on the model, favour interpretable models "
             "like linear regression or a shallow tree, or pair a strong model with "
             "explanation tools. A slightly less accurate model people understand often "
             "beats a black box."},

    # --- specialised tasks ---
    {"id": "timeseries", "title": "Time-series forecasting",
     "tags": "time series forecasting lag seasonality trend",
     "text": "Do not shuffle time-series data, split by time. Trees cannot extrapolate "
             "a trend, so model the differenced series or add lag, rolling, and calendar "
             "features like day of week and month. Watch for seasonality and holidays."},
    {"id": "text-nlp", "title": "Text classification",
     "tags": "text nlp tfidf embeddings stopwords",
     "text": "Vectorise text with TF IDF over unigrams and bigrams before a linear or "
             "tree model, or use embeddings for richer meaning. Clean and lower case the "
             "text, remove stop words, and beware leakage from duplicated documents."},
    {"id": "clustering-k", "title": "Choosing the number of clusters",
     "tags": "clustering kmeans elbow silhouette k",
     "text": "For KMeans, pick the number of clusters with the elbow of the inertia "
             "curve or the best silhouette score, and always check the clusters make "
             "business sense. Density methods like DBSCAN find the count themselves and "
             "can flag outliers."},

    # --- interpretation and next steps ---
    {"id": "feature-importance", "title": "Reading feature importance",
     "tags": "importance coefficients shap permutation",
     "text": "Tree importances and linear coefficients rank predictive features but can "
             "be skewed by correlation and scale. Permutation importance and SHAP give "
             "more trustworthy attributions and can explain a single prediction."},
    {"id": "eda", "title": "Exploratory data analysis",
     "tags": "eda distributions correlation target",
     "text": "Before modelling, inspect distributions, missingness, correlations, and "
             "the target's balance or skew. Strongly correlated features may be "
             "redundant, and skew or heavy tails may call for a transform."},
    {"id": "correlation", "title": "Correlation is not causation",
     "tags": "correlation causation confounder",
     "text": "A strong correlation between two columns does not mean one causes the "
             "other. A hidden third factor may drive both. Use correlations to guide "
             "features and questions, not to claim cause."},
    {"id": "tuning", "title": "Hyperparameter tuning",
     "tags": "tuning grid search cross validation",
     "text": "Search hyperparameters with cross validation so you do not tune to one "
             "lucky split. Start with a small grid around sensible defaults. Tuning "
             "gives real but usually modest gains over a good default."},
    {"id": "ensembling", "title": "Ensembling models",
     "tags": "ensemble voting stacking blend",
     "text": "Combining several different models, by voting or stacking, often beats any "
             "single one because their errors partly cancel. It works best when the "
             "models are strong and make different kinds of mistake."},
    {"id": "baselines", "title": "Always set a baseline",
     "tags": "baseline reference sanity",
     "text": "Compare every model against a trivial baseline, the majority class for "
             "classification or the mean for regression. If a model cannot beat that, "
             "the features do not carry enough signal yet."},
    {"id": "deployment", "title": "From model to use",
     "tags": "deployment monitoring drift predict",
     "text": "A trained model is only useful once it scores new data. Save the whole "
             "pipeline, feed it new rows with the same columns, and monitor for drift as "
             "the world changes, since a model trained on old patterns slowly goes "
             "stale."},
]
