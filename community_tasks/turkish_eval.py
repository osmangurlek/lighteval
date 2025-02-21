from lighteval.src.lighteval.metrics.metrics import LoglikelihoodAcc
from lighteval.src.lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.src.lighteval.tasks.requests import Doc

def prompt_fn_turkish_eval_task(line, task_name: str = None):
    """
    Prepares a question and answer set in Turkish from the Turkish Evaluation DatasetArgs:
        line (dict): A dictionary with required keys:
            - 'query' (str): The main question string.
            - 'choices' (list of str): A list containing exactly five answer options.
            - 'answer_str' (str): A single character from "A" to "E" representing the correct answer.
        task_name (str, optional): An optional string specifying the evaluation task name.

    Returns:
        Doc: A structured object for LightEval containing:
            - task_name (str): The task name, if provided.
            - query (str): Formatted question with embedded answer choices.
            - choices (list of str): List of option identifiers ["A", "B", "C", "D", "E"].
            - gold_index (int): Index of the correct answer within the 'choices' list.

    Raises:
        ValueError: If the 'choices' list does not contain exactly five items,
                    or if 'answer_str' is not one of ["A", "B", "C", "D", "E"].
    """
    
    query_template = """Lütfen aşağıdaki metne dayanarak soruyu yanıtlayın. Yalnızca verilen seçeneklerden doğru olanı (A, B, C, D veya E) belirtin; ek açıklama yapmayın.
    Metin: {narrative}
    Soru: {question}

    Seçenekler:
    A) {choice_a}
    B) {choice_b}
    C) {choice_c}
    D) {choice_d}
    E) {choice_e}
    
    Cevap:
    """

    options = line["choices"]

    query = query_template.format(
        narrative=line["narrative"],
        question=line["question"],
        choice_a=options[0],
        choice_b=options[1],
        choice_c=options[2],
        choice_d=options[3],
        choice_e=options[4],
    )

    choices = ["A", "B", "C", "D", "E"]
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=choices.index(line["answer_choice"]),
    )
    
task = LightevalTaskConfig(
    name="turkish_eval_complex_ales",
    prompt_function=prompt_fn_turkish_eval_task,
    hf_repo="metunlp/complex-ales",
    metric=[LoglikelihoodAcc],
)