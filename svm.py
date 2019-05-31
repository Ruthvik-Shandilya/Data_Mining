import numpy as np

class SVM(object):
    def fit(self,data):
        self.data=data

        opt_dict = {}

        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]

        all_data = np.array([])
        for yi in self.data:
            all_data = np.append(all_data, self.data[yi])

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data=None


        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01]

        # extremly expensise
        b_range_multiple = 5
        # we dont need to take as small step as w
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        # making step smaller and smaller to get precise value
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])

            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1 * self.max_feature_value * b_range_multiple,
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True

                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                        if found_option:
                            """
                            all points in dataset satisfy y(w.x)+b>=1 for this cuurent w_t, b
                            then put w,b in dict with ||w|| as key
                            """
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                # after w[0] or w[1]<0 then values of w starts repeating itself because of transformation
                # Think about it, it is easy
                # print(w,len(opt_dict)) Try printing to understand
                if w[0] < 0:
                    optimized = True
                    print("optimized a step")
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]

            latest_optimum = opt_choice[0][0] + step * 2

    def predict(self,features):

        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification!=0:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return (classification, np.dot(np.array(features), self.w) + self.b)




