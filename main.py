import Dao_email
import Naive_Bayse
import mail_extraction
import pandas
from Naive_Bayse import split
from Naive_Bayse import morphs
from Naive_Bayse import noun


def get_result(method):
    train_set, test_set = mail_extraction.making_doclist(0.8, Dao_email.connection_sqlite)

    q, w, e = Naive_Bayse.training(["True", "False"], train_set, method)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    result, target = Naive_Bayse.testing_all(q, w, e, test_set, method)

    output = Naive_Bayse.compare_result(result, target)

    return output


def make_average(cnt, method):
    total = pandas.array([0] * 4)
    for _ in range(cnt):
        total = total + pandas.array(get_result(method))
        print("----------------------------------")

    result = total / cnt

    print("--------Total-------------")
    print("precision:" + str(result[0]))
    print("accuracy:" + str(result[1]))
    print("Recall:" + str(result[2]))
    print("F1-score:" + str(result[3]))
    return result


make_average(10, Naive_Bayse.split)
make_average(10, Naive_Bayse.morphs)
make_average(10, Naive_Bayse.noun)
