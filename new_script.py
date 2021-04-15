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
    alg = 1
    dataset = "ricci"
    is_accuracy = False
    type = "random_oversampled"
    count = 10

    for tax in ["S-0.0-B-0.6-R-0.4"]:
        for alpha in ["1.4"]:
            ADIs = []
            ACCs = []
            for run in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
                filename = "{}/test-size-0.2/taxonomy-weights-{}/oversampling-factor-{}/run-{}/{}_results.txt".format(dataset, tax, alpha, run, type)
                results = get_results(filename)

                if dataset == "credit":
                    personal_status_disparate_impact = results.iloc[alg]["age_disparate_impact"]
                    ADI_personal_status = personal_status_disparate_impact
                    if ADI_personal_status > 1:
                        ADI_personal_status = 1 / ADI_personal_status

                    age_disparate_impact = results.iloc[alg]["personal_status_disparate_impact"]
                    ADI_age = age_disparate_impact
                    if ADI_age > 1:
                        ADI_age = 1 / ADI_age

                    ADI_run = (ADI_personal_status + ADI_age) / 2

                elif dataset == "ricci":
                    ADI_run = results.iloc[alg]["Race_disparate_impact"]

                    if ADI_run > 1:
                        ADI_run = 1 / ADI_run

                """
                if np.isnan(ADI_run):
                    count -= 1
                else:"""
                ADIs.append(ADI_run)
                ACCs.append(results.iloc[alg]["accuracy"])

            if is_accuracy:
                print(round(sum(ACCs) / count, 2))
            else:
                print(round(sum(ADIs) / count, 2))
