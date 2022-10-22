import Naive_Bayse
import mail_extraction
from Naive_Bayse import split
from Naive_Bayse import morphs
from Naive_Bayse import noun

train_set, test_set = mail_extraction.making_doclist(0.8)

q, w, e = Naive_Bayse.training([True, False], train_set, split())

result, target = Naive_Bayse.testing_all(q, w, e, test_set, split())

Naive_Bayse.compare_result(result, target)
