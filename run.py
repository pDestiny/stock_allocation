import argparse
import datetime as dt
from mptfunc import get_data, max_sharp_ratio, minimize_variance, calculated_results, plot_data, save_efficient_list
from pathlib import Path
import logging
import sys

# required arguments for the whole program
parser = argparse.ArgumentParser(description="""
------MPT based asset allocation model-----
Output List(without --cal option)
1. efficient_frontier_{start}_{end}_{days}d.xls
    efficient frontier result excel file.
2. efficient_frontier_{start}_{end}_{days}d.png
    efficient frontier result plot image.
""")
parser.add_argument("--stock-ids", metavar="N", required=True, type=str, nargs="+", help="**Korea** stock ids. This program is implemented only for Korea stocks. Ex) '--stock-ids 005930 000660' this means the program will optimize allocation with Samsung Eletronics and SK hynix stock")
parser.add_argument("--start", type=lambda x: dt.datetime.strptime(x, "%Y-%m-%d"), required=True, dest="start", help="The start date(ex. 2023-07-20) where the investor wants to analyze. It should be prior to end")
parser.add_argument("--end", type=lambda x: dt.datetime.strptime(x, "%Y-%m-%d"), required=True, dest="end", help="The end date(ex. 2024-07-20) where the investor wants to analyze. It should be later from start")
parser.add_argument("--days", type=int, required=True, help="The period for calculating expected return and standard deviation. Ex. '--days 20' this means your expected return and standard deviation will be multiplied by 20 from one day based expected return and standard devaition")
parser.add_argument("--outdir", type=Path, default="./output", help="output folder")
parser.add_argument("--resolution", type=int, default=50, help="this option determine how granular target return will be. default is 50")
args = parser.parse_args()

logger = logging.getLogger("MDP Logger")

handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def main():
    try:
        # check data validation
        assert args.outdir.exists(), "Your output directory doesn't exist"
        assert args.start < args.end, "Start date should be forward from end date"
        logger.info("Download Korea Stock Data Started")
        # download data
        mean_return, cov_matrix = get_data(args.stock_ids, args.start, args.end)
        logger.info("Download Korea Stock Data Finished")
        # calculate max sharp ratio
        logger.info("Calculating Max Sharp Ratio Started")

        max_sharp_ratio_result = max_sharp_ratio(mean_return, cov_matrix, args.days)

        logger.info("Calculating Max Sharp Ratio Finished")
        # calculate minimize variance
        logger.info("Calculating Min Variance Optimization Started")
        min_var_result = minimize_variance(mean_return, cov_matrix, args.days)
        logger.info("Calculating Min Variance Optimization Finished")
        # calculate efficient frontier
        logger.info("Calculating Efficient Frontier Started")

        max_sharp_point, min_vol_point, efficient_df = \
            calculated_results(max_sharp_ratio_result, min_var_result, mean_return, cov_matrix, args.days, args.resolution)

        logger.info("Calculating Efficient Frontier Finished")

            # output result to output folder
        logger.info("Generating Output Started")
        plot_data(max_sharp_point, min_vol_point, efficient_df, args.outdir, args.start, args.end, args.days)
        save_efficient_list(args.stock_ids, efficient_df, args.outdir, args.start, args.end, args.days)
        logger.info("Generating Output Finished")

    except Exception as e:
        logger.error(e)

if __name__ == "__main__":
    main()
    exit(0)



