import copy
import numpy as np
import utils.transform as transform


class GlobalRegistrationVerification:
    
    def __init__(self):
        self.local_t = []
        self.global_t = []
        self.sequence_ts = []
        self.global_inds = []
        self.global_target_t = []
        self.found_correct_global = False
        self.found_correct_global_at = -1
        
    def update_local(self, local_timestamp, local_transformation):
        self.sequence_ts.append(local_timestamp)
        self.local_t.append(local_transformation)
        
        if self.found_correct_global:
            self.global_t.append(np.dot(self.global_t[-1], local_transformation))
        else:
            self.global_t.append(np.identity(4))
        
    
    def update_global(self, global_timestmap, global_transformation):
        index = np.argwhere(np.array(self.sequence_ts) == global_timestmap).flatten()
        
        if len(index) == 0:
            print(f"Timestamp {global_timestmap} not found in sequence.")
        else:
            index = index[0]
        
        self.global_inds.append(index)
        self.global_target_t.append(global_transformation)
        
        if len(self.global_inds) > 2 and not self.found_correct_global:
            self.verify()

    @staticmethod
    def validate(T1, T2, T3, t1, t2, max_dist, max_rot):
        c1 = transform.check(T3, np.dot(T2, t2), max_t=max_dist, max_r=max_rot)
        c2 = transform.check(T3, np.dot(np.dot(T1, t1), t2), max_t=max_dist, max_r=max_rot)
        c3 = transform.check(T2, np.dot(T1, t1), max_t=max_dist, max_r=max_rot)

        print(f"Check 1: {c1}, Check 2: {c2}, Check 3: {c3}")
        
        # If two checks are true, the combination is wrong
        if (c1 + c2 + c3) == 2:
            raise Exception("Invalid combination")

        # If two checks are true, the combination is wrong
        if (c1 + c2 + c3) == 0:
            raise Exception("Invalid transformations")

        # If all the checks are valid, there is no need of correction
        if c1 and c2 and c3:
            print(":: No need of correction.")
            return T1, T2, T3
        
        # If two checks are wrong, only one transformation needs correction
        if c1:
            # print(":: Correcting Previous Transformation")
            T1 = np.dot(T2, transform.inv_transform(t1))
        elif c2:
            # print(":: Correcting Current Transformation")
            T2 = np.dot(T1, t1)
        else:
            # print(":: Correcting Future Transformation")
            T3 = np.dot(T2, t2)

        return T1, T2, T3

    @staticmethod
    def merge_transformation_matrices(start_t, end_t, local_t):
        local_ts = np.identity(4)

        for t in range(start_t, end_t):
            local_ts = np.dot(local_t[t + 1], local_ts)
            
        return local_ts
            
    
    def verify(self):
        for t in range(len(self.global_inds)):
            if t > 1:
                print(f"Global registration verification: {t}/{len(self.global_inds)}")
                total = 0
                for i in range(t, t - 3, -1):
                    if np.sum(self.global_target_t[i]) == 4:
                        total += 1
                        
                print(f"Total invalid global registrations: {total}")        
                if total > 1: return
                
                print(f"Validating and correcting global registrations.")
                try:
                    self.global_target_t[t - 2], self.global_target_t[t - 1], self.global_target_t[t] = GlobalRegistrationVerification.validate(
                        self.global_target_t[t - 2], self.global_target_t[t - 1], self.global_target_t[t], 
                        GlobalRegistrationVerification.merge_transformation_matrices(self.global_inds[t - 2], self.global_inds[t - 1], self.local_t),
                        GlobalRegistrationVerification.merge_transformation_matrices(self.global_inds[t - 1], self.global_inds[t], self.local_t),
                        max_rot=5, max_dist=0.3
                    )
                    self.found_correct_global = True
                    self.found_correct_global_at = t
                except Exception as e:
                    print(f"Exception:", e)
                    continue
        
        if self.found_correct_global:
            self.global_t[self.global_inds[self.found_correct_global_at]] = self.global_target_t[self.found_correct_global_at]

            for t in range(self.global_inds[self.found_correct_global_at] + 1, len(self.global_t)):
                self.global_t[t] = np.dot(self.global_t[t - 1], self.local_t[t])
                
            for t in range(self.global_inds[self.found_correct_global_at] - 1, -1, -1):
                self.global_t[t] = np.dot(self.global_t[t + 1], transform.inv_transform(self.local_t[t + 1]))

    def get_global_t(self):
        return copy.deepcopy(self.global_t)