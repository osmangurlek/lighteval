from lighteval.metrics.metrics import LoglikelihoodAcc
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
# Özel kayıt için alternatif fonksiyon denemesi
try:
    from lighteval.tasks.registry import register_task
except ImportError:
    register_task = None

def prompt_fn_turkish_eval_task(line, task_name: str = None):
    """
    Türkçe Değerlendirme Görevi için soru ve cevap seti hazırlar.
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
    
    Cevabınız:"""

    options = line["choices"]
    if len(options) != 5:
        raise ValueError("Seçenek listesi tam olarak 5 öğe içermelidir.")

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
    if line["answer_choice"] not in choices:
        raise ValueError("answer_choice değeri geçerli değil; A, B, C, D veya E olmalıdır.")

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=choices.index(line["answer_choice"]),
    )
    
task = LightevalTaskConfig(
    name="turkish_eval",
    prompt_function=prompt_fn_turkish_eval_task,
    suite=["community"],
    hf_repo="metunlp/complex-ales",
    hf_subset="default",
    hf_avail_splits=[],
    evaluation_splits=[],
    few_shots_split=None,
    few_shots_select=None,
    metric=[LoglikelihoodAcc],
)

# Eğer register_task mevcutsa kullanmayı deneyin, yoksa dosyayı import etmek özel görevi otomatik kaydediyor olabilir.
if register_task is not None:
    register_task(task)

# Artık lighteval accelerate komutuyla görevinizi çağırabilirsiniz:
# lighteval accelerate "pretrained=gpt2" "community|turkish_eval|0|0"