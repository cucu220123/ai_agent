"""Recovery must never change already measured evidence."""
import hashlib
import json
from types import SimpleNamespace
from app.models import CapabilitySpec, KnowledgeContext
from scripts.acceptance_recovery import recover_explanation


def test_explanation_recovery_preserves_measured_run_and_original_report(tmp_path):
    calls=[]
    class Explainer:
        def run(self, *args):
            calls.append({'call_id':'llm_0001','purpose':'explanation','provider':'fixture','status':'ok'})
            return {'status':'ok','why_this_plan':'Fixture explanation'}
    result={'run_id':'abc','report_json':str(tmp_path/'report.json'),'explanation':{'status':'deterministic_fallback'},'spec':CapabilitySpec(raw_description='test').to_dict(),'knowledge':KnowledgeContext().to_dict(),'candidate_results':[{'plan':{'algorithm_id':'candidate'},'validation':{'metrics':{'roc_auc':.87},'status':'passed'}}],'selected_plan':{'algorithm_id':'candidate'},'validation':{'metrics':{'roc_auc':.87},'status':'passed'},'writeback':{'validation_run_ids':['abc']},'generated_files':['immutable-v1.py'],'llm_trace':{'calls':[]}}
    repaired=recover_explanation(result,SimpleNamespace(llm=SimpleNamespace(calls=calls),explainer=Explainer()),tmp_path)
    prior_path=tmp_path/'report_revisions/abc.before-explanation.json'
    original=json.loads(prior_path.read_text())
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==repaired['report_revision']['previous_report_sha256']
    for field in repaired['report_revision']['preserved_fields']:
        assert repaired[field]==original[field]
    assert original['explanation']['status']=='deterministic_fallback'
    assert repaired['llm_trace']['calls'][-1]['call_id']=='explanation_recovery/llm_0001'
