import pytest
from app.agents.explanation_agent import validate_comparative_claims


def test_explanation_cannot_reverse_measured_auc_order():
    candidates = [
        {'plan': {'algorithm_name': 'Logistic Regression'}, 'validation': {'metrics': {'roc_auc': 0.8655}}},
        {'plan': {'algorithm_name': 'Gradient Boosting'}, 'validation': {'metrics': {'roc_auc': 0.8554}}},
    ]
    bad = {"limitations": ["Logistic Regression's performance is slightly lower than Gradient Boosting in terms of ROC-AUC."]}
    with pytest.raises(ValueError, match='contradictory comparison'):
        validate_comparative_claims(bad, candidates)
    good = {"limitations": ["Logistic Regression has higher ROC-AUC than Gradient Boosting."]}
    validate_comparative_claims(good, candidates)


def test_explanation_metric_direction_and_unmeasured_comparison():
    candidates = [
        {'plan': {'algorithm_name': 'Ridge'}, 'validation': {'metrics': {'mae': 0.1}}},
        {'plan': {'algorithm_name': 'Forest'}, 'validation': {'metrics': {'mae': 0.2}}},
    ]
    validate_comparative_claims({'limitations': ['Ridge has better MAE than Forest.']}, candidates)
    with pytest.raises(ValueError, match='unmeasured comparison'):
        validate_comparative_claims({'limitations': ['Ridge has higher ROC-AUC than Forest.']}, candidates)



def test_historical_ranking_requires_evidence_beyond_current_candidates():
    current=[{'plan':{'algorithm_name':'Gradient Boosting'},'validation':{'metrics':{'roc_auc':.88}}}]
    with pytest.raises(ValueError,match='historical rankings'):
        validate_comparative_claims({'why_this_plan':'Historical evidence shows Gradient Boosting outperforming Logistic Regression.'},current)


def test_unchecked_historical_failures_are_rejected_but_prior_citations_allowed():
    with pytest.raises(ValueError, match='historical outcome'):
        validate_comparative_claims({'limitations': ['Previous runs of Logistic Regression have failed.']}, [])
    validate_comparative_claims({'why_this_plan': 'Historical cases cited below informed uncertain planning priors. Current validation determines the winner.'}, [])
