from evaluation.vqaEval import VQAEval
from evaluation.vqa import VQA

def Evaluate(ans_file_path, ques_file_path, result_eval_file):
    vqa = VQA(ans_file_path, ques_file_path)
    vqaRes = vqa.loadRes(result_eval_file, ques_file_path)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate()

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
    # print("Per Question Type Accuracy is the following:")
    # for quesType in vqaEval.accuracy['perQuestionType']:
    #     print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
    # print("\n")
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")