import datetime
from code.reward import calc_global_reward, calc_difference_reward, calc_dpp_reward, calc_sdpp_reward # Agent Reward


class Mod:

    def global_reward_mod(data):
        data["Mod Name"] = "global"

        date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
        print("Starting %s test at\n\t%s"%(data["Mod Name"], date_time_string))

        # Agent Reward
        data["Reward Function"] = calc_global_reward

        data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

    def difference_reward_mod(data):
        data["Mod Name"] = "difference"

        date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
        print("Starting %s test at\n\t%s"%(data["Mod Name"], date_time_string))

        # Agent Reward
        data["Reward Function"] = calc_difference_reward

        data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

    def dpp_reward_mod(data):
        data["Mod Name"] = "dpp"

        date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
        print("Starting %s test at\n\t%s"%(data["Mod Name"], date_time_string))

        # Agent Reward
        data["Reward Function"] = calc_dpp_reward

        data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

    def sdpp_reward_mod(data):
        data["Mod Name"] = "s-dpp"

        date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
        print("Starting %s test at\n\t%s" % (data["Mod Name"], date_time_string))

        # Agent Reward
        data["Reward Function"] = calc_sdpp_reward

        data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv" % \
                                             (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv" % \
                                            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle" % \
                                        (data["Specifics Name"], data["Mod Name"], date_time_string)
