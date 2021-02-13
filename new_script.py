import re
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
import pandas as pd


def get_results(filename):
    f = open(filename, "r")
    text = f.read()
    text = re.sub(r"{[^{}]+}", lambda x: x.group(0).replace(",", ";"), text)
    f.close()

    return pd.read_csv(StringIO(text), sep=',', engine='python')


if __name__ == '__main__':

    for alpha in ["0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4"]:

        ADIs = []
        for run in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
            filename = "credit/test-size-0.3/taxonomy-weights-S-0.0-B-0.6-R-0.4/oversampling-factor-{}/run-{}/oversampled_results.txt".format(alpha, run)
            results = get_results(filename)
            personal_status_disparate_impact = results.iloc[4]["accuracy"]
            age_disparate_impact = results.iloc[0]["accuracy"]

            ADI_run = (age_disparate_impact + personal_status_disparate_impact) / 2
            if ADI_run > 1:
                ADI_run = 1 / ADI_run
            ADIs.append(ADI_run)

        print(round(sum(ADIs)/10, 3))
    """
    for alpha in ["0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4"]:

        ADIs = []
        for run in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
            #filename = "ricci/test-size-0.3/taxonomy-weights-S-0.0-B-0.4-R-0.6/oversampling-factor-{}/run-{}/oversampled_results.txt".format(alpha, run)
            filename = "ricci/test-size-0.3/taxonomy-weights-S-0.0-B-0.6-R-0.4/oversampling-factor-{}/run-{}/oversampled_results.txt".format(alpha, run)
            #filename = "ricci/test-size-0.3/taxonomy-weights-S-0.0-B-0.5-R-0.5/oversampling-factor-{}/run-{}/oversampled_results.txt".format(alpha, run)
            #filename = "ricci/test-size-0.3/taxonomy-weights-S-0.33-B-0.33-R-0.33/oversampling-factor-{}/run-{}/oversampled_results.txt".format(alpha, run)

            results = get_results(filename)
            personal_status = results.iloc[2]["accuracy"]

            ADI_run = personal_status
            if ADI_run > 1:
                ADI_run = 1 / ADI_run
            ADIs.append(ADI_run)

        print(round(sum(ADIs)/len(ADIs), 2))"""
