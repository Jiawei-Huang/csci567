import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    numerator = np.array(real_labels) * np.array(predicted_labels)
    numerator = sum(numerator)

    denominator = sum(real_labels) + sum(predicted_labels)
    if denominator == 0:
        return 0

    f1_score = 2 * numerator/denominator
    
    return f1_score
    raise NotImplementedError




class Distances:
    @staticmethod
    # TODO
    def canberra_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
  
        point1 = np.array(point1)
        point2 = np.array(point2)

        numerator = point1 - point2
        numerator = abs(numerator)

        denominator = abs(point1) + abs(point2)
        denominator[denominator==0] = 1

        dist = sum(numerator/denominator)

        return dist
        raise NotImplementedError
      
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        point1 = np.array(point1)
        point2 = np.array(point2)

        numerator = abs(point1 - point2)
        numerator = numerator**3

        dist = np.sign(sum(numerator)) * (np.abs(sum(numerator))) ** (1 / 3)

        return dist
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        point1 = np.array(point1)
        point2 = np.array(point2)

        numerator = point1 - point2
        numerator = numerator**2

        dist = sum(numerator)**(1/2)

        return dist
        raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.dot(point1, point2)
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        point1 = np.array(point1)
        point2 = np.array(point2)
        point1_len = np.sqrt(sum(point1**2))
        point2_len = np.sqrt(sum(point2**2))


        numerator = point1 - point2
        numerator = numerator**2

        inner_dist = np.dot(point1, point2)
        if point1_len * point2_len == 0:
            dist = 1
        else:
            dist = 1 - inner_dist/(point1_len * point2_len)

        return dist
        raise NotImplementedError

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        point1 = np.array(point1)
        point2 = np.array(point2)
        inner = np.dot(point1-point2, point1-point2)
        dist = -np.exp(-0.5 * inner)
        return dist
        raise NotImplementedError


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        dist_func_whole = ["canberra", "minkowski", "euclidean", "gaussian", "inner_prod", "cosine_dist"]
        dist_func = [i for i in dist_func_whole if i in distance_funcs.keys()]

        results = []
        models = []

        for name in dist_func:
            result = []
            for k in range(1, 30, 2):
                func = distance_funcs[name]
                knn = KNN(k, func)
                knn.train(x_train, y_train)
                models.append(knn)
                label = knn.predict(x_val)
                f1 = f1_score(y_val, label)
                result.append(f1)
            results.append(result)
        
        results = np.array(results)
        index = np.argmax(results)
        row_ind, col_ind = np.unravel_index(index, results.shape)

        # row_ind = index // results.shape[1]
        # col_ind = index % results.shape[1]
        
        # You need to assign the final values to these variables
        self.best_k = col_ind * 2 + 1
        self.best_distance_function = dist_func[row_ind]
        self.best_model = models[index]
        # raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        # Method 1
        # best_f1_score, best_k = 0, -1
        # for scaling_name, scaling_class in scaling_classes.items():
        #     for name, func in distance_funcs.items():
        #         scaler = scaling_class()
        #         train_features_scaled = scaler(x_train)
        #         valid_features_scaled = scaler(x_val)
        #         k_lim = len(x_train) - 1
        #         for k in range(1, min(31, k_lim), 2):
        #             model = KNN(k=k, distance_function=func)
        #             model.train(train_features_scaled, y_train)
        #             valid_f1_score = f1_score(y_val, model.predict(valid_features_scaled))
        #             if valid_f1_score > best_f1_score:
        #                 best_f1_score, best_k = valid_f1_score, k
        #                 best_model = model
        #                 best_func = name
        #                 best_scaler = scaling_name

        # self.best_k = best_k
        # self.best_distance_function = best_func
        # self.best_scaler = best_scaler
        # self.best_model = best_model


        dist_func_whole = ["canberra", "minkowski", "euclidean", "gaussian", "inner_prod", "cosine_dist"]
        dist_func = [i for i in dist_func_whole if i in distance_funcs.keys()]

        scaler_whole = ["min_max_scale", "normalize"]
        scaler = [i for i in scaler_whole if i in scaling_classes.keys()]

        results = []
        models = []

        # loop through different scaling classes
        for class_name in scaling_classes.keys():
            result_scale = []
            scale = scaling_classes[class_name]()
            x_train_scaled = scale(x_train)
            x_val_scaled = scale(x_val)
            # loop through different k
            for name in distance_funcs.keys():
                # loop through distance functions
                result_func = []
                for k in range(1, 30, 2):
                    func = distance_funcs[name]
                    knn = KNN(k, func)
                    knn.train(x_train_scaled, y_train)
                    models.append(knn)
                    # scalers.append(scale)
                    label = knn.predict(x_val_scaled)
                    f1 = f1_score(y_val, label)
                    result_func.append(f1)

                result_scale.append(result_func)

            results.append(result_scale)

        
        results = np.array(results)
        index = np.argmax(results)
        # scale_ind = index // (results.shape[1] * results.shape[2])
        # scale_ind = index // (results.shape[1] * results.shape[2])
        # rest = index % (results.shape[1] * results.shape[2])
        # row_ind = rest // results.shape[2]
        # col_ind = rest % results.shape[2]
        scale_ind, func_ind, k_ind = np.unravel_index(index, results.shape)

        
        # You need to assign the final values to these variables
        self.best_k = k_ind * 2 + 1
        self.best_distance_function = list(distance_funcs.keys())[func_ind]
        self.best_scaler = list(scaling_classes.keys())[scale_ind]
        self.best_model = models[index]


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normal = np.linalg.norm(features, axis=1).reshape((-1, 1))
        normal[normal==0] = 1
        result = np.array(features)/normal
        # result = np.nan_to_num(result, nan=0)
        return result.tolist()

        raise NotImplementedError


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        pass

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.array(features)
        try:
            if self.__class__.__call__.called:
                result = (features - self.col_min)/(self.col_max - self.col_min)
                return result.tolist()
        except AttributeError:
            self.__class__.__call__.called = True
            self.col_min = np.min(features, axis=0).reshape((1, -1))
            self.col_max = np.max(features, axis=0).reshape((1, -1))
            # if min == max, set corresponding min=0, max=1
            match = self.col_max == self.col_min
            self.col_min[match] = 0
            self.col_max[match] = 1

            result = (features - self.col_min)/(self.col_max - self.col_min)

            return result.tolist()

        raise NotImplementedError