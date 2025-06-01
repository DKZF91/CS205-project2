import numpy as np


def evaluate_accuracy(features, label, feature_i):
    features_s = features[:, feature_i]  # Select the new feature subset composed of feature_i columns
    correct = 0
    n = len(features_s)

    for i in range(n):
        test_sample = features_s[i]  # The current set of test samples
        test_label = label[i]

        data_train = np.delete(features_s, i, axis=0)  # Delete this sample and use it as a training set
        label_train = np.delete(label, i)

        distances = np.linalg.norm(data_train - test_sample, axis=1)  # Calculate Euclidean distance
        nearest_index = np.argmin(distances)
        predicted_label = label_train[nearest_index]

        if predicted_label == test_label:
            correct += 1

    return correct / n


def forward_selection(features, label):
    n_features = features.shape[1]
    features_s = []  # Currently selected feature set
    best_ac = 0  # Current best accuracy
    history = []  # Store the subset and corresponding accuracy of each round

    for _ in range(n_features):  # Choose at most n rounds
        best_candidate = None
        for i in range(n_features):
            if i not in features_s:
                candidate = features_s + [i]
                ac = evaluate_accuracy(features, label, candidate)
                history.append((list(candidate), ac))
                if ac > best_ac:
                    best_ac = ac
                    best_candidate = i
        if best_candidate is not None:
            features_s.append(best_candidate)
        else:  # If there is no more improvement, end early
            break
    return features_s, best_ac, history


def backward_elimination(features, label):
    selected_features = list(range(features.shape[1]))  # Initially include all features
    history = []

    best_subset = list(selected_features)
    best_accuracy = evaluate_accuracy(features, label, selected_features)
    history.append((list(selected_features), best_accuracy))

    while len(selected_features) > 1:  # Try to remove one feature in each round and select the subset with the largest accuracy improvement after removal
        best_candidate_subset = None
        best_candidate_accuracy = -1

        for i in selected_features:
            candidate = [f for f in selected_features if f != i]
            acc = evaluate_accuracy(features, label, candidate)
            history.append((list(candidate), acc))
            if acc > best_candidate_accuracy:
                best_candidate_accuracy = acc
                best_candidate_subset = candidate

        selected_features = best_candidate_subset

        if best_candidate_accuracy > best_accuracy:  # Current best accuracy
            best_accuracy = best_candidate_accuracy
            best_subset = list(best_candidate_subset)

    return best_subset, best_accuracy, history


def feature_selection(filepath, method):
    data = np.loadtxt(filepath)
    features = data[:, 1:]
    label = data[:, 0].astype(int)

    print(f"\nThis dataset has {features.shape[1]} features (not including the class attribute), with {len(label)} instances.")
    acc_all = evaluate_accuracy(features, label, list(range(features.shape[1])))
    print(f"\nRunning nearest neighbor with all {features.shape[1]} features, using 'leaving-one-out' evaluation, I get an accuracy of {acc_all * 100:.1f}%")

    if method == 1:
        print("\nBeginning forward selection search...")
        best_features, best_acc, history = forward_selection(features, label)
    elif method == 2:

        print("\nBeginning backward elimination search...")
        best_features, best_acc, history = backward_elimination(features, label)
    else:
        print("Invalid method selected.")
        return

    for features, acc in history:
        print(f"Using feature(s) {[i+1 for i in features]} accuracy is {acc * 100:.1f}%")

    print(f"\nFinished search!! The best feature subset is {[i+1 for i in best_features]}, which has an accuracy of {best_acc * 100:.1f}%.")


if __name__ == "__main__":
    print("Welcome to my Feature Selection Algorithm.")
    file_name = input("Type in the name of the file to test : ").strip()
    # CS205_large_Data__49.txt
    # CS205_small_Data__38.txt

    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection\n2) Backward Elimination")

    method = int(input().strip())
    feature_selection(file_name, method)

