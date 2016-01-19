# Parallelized top-k leave-one-out cross-validation
# Trivially parallelizable, so simply saturate as many cores as desired with processes which 
# get the accuracy of top-k features
from multiprocessing import Process, Queue
import sklearn
 
# Results will be stored in the following dictionary as {k: accuracy}   
accuracies = {}
# ranked_features should be a list of feature names sorted by decreasing importance (e.g. using random forests or f_classif)
ranked_features = feature_importances["feature"].values
# Largest feature subset
max_k = len(ranked_features)
# How many cores to saturate
cores = 8

counter = 0
procs_and_queues = []

def proc(id, q, cores, ranked_features, feature_values, class_labels, max_k):
    lrc = sklearn.linear_model.LogisticRegression()
    n = len(class_labels)
    loo = sklearn.cross_validation.LeaveOneOut(n=n)
    for k in range(id, max_k, cores):
        features = ranked_features[:k]
        X = np.array(feature_values[features])
        n_correct = 0
        for train_index, test_index in loo:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = class_labels[train_index], class_labels[test_index]
            lrc.fit(X_train, y_train)
            y_prediction = lrc.predict(X_test)[0]
            if (y_test == y_prediction):
                n_correct = n_correct + 1
        accuracy = float(n_correct) / n
        q.put([k, accuracy])
    q.put(None)

for i in range(cores):
    q = Queue()
    p = Process(target=proc, args=(i + 1, 
                                   q, 
                                   cores, 
                                   ranked_features, 
                                   X,
                                   np.array(class_labels),
                                   max_k))
    procs_and_queues.append([p, q])
    
for p, q in procs_and_queues:
    p.start()
    
#do the processing and collect the results
open_procs = cores
while open_procs > 0:
    for p, q in procs_and_queues:
        new_item = q.get()
        if new_item is None:
            print("closing")
            p.join()
            open_procs = open_procs - 1
        else:
            k, accuracy = new_item
            accuracies[k] = accuracy
            counter = counter + 1
            print(counter)
            if (counter % 50) == 0:
                print(str(counter) + "/" + str(max_k) + " - " + str(100.0 * (float(counter) / max_k)) + "% complete")
