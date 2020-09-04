seed = 6

models = {'LogReg': LogisticRegression(),
          'KNN': KNeighborsClassifier(),
          'DT': DecisionTreeClassifier(random_state = seed), 
          'GaussianNB': GaussianNB(),
          'MultinomailNB': MultinomialNB(),
          'LDA': LinearDiscriminantAnalysis(),
          'LinearSVC': LinearSVC(max_iter = 1250, random_state = seed),
          'SGD': SGDClassifier(random_state = seed),  
          'ADA': AdaBoostClassifier(random_state = seed),
          'Bagging': BaggingClassifier(random_state = seed), 
          'Ridge': RidgeClassifier(random_state = seed),
          'RF': RandomForestClassifier(random_state = seed),
          'GradientBoost' : GradientBoostingClassifier(random_state = seed)}