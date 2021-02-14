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
    type = "train"

    for tax in ["S-0.0-B-0.4-R-0.6", "S-0.0-B-0.5-R-0.5", "S-0.33-B-0.33-R-0.33", "S-0.0-B-0.6-R-0.4"]:
        for alpha in ["0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4"]:
            ADIs = []
            ACCs = []
            for run in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
                filename = "{}/test-size-0.3/taxonomy-weights-{}/oversampling-factor-{}/run-{}/{}_results.txt".format(dataset, tax, alpha, run, type)
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

                ADIs.append(ADI_run)
                ACCs.append(results.iloc[alg]["accuracy"])

            if is_accuracy:
                print(round(sum(ACCs) / 10, 2))
            else:
                print(round(sum(ADIs) / 10, 2))
