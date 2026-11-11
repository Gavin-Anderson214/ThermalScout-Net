from scripts.diagnose_padding import content_rect, inside, human_overlap


def test_letterbox_rectangle_matches_integer_resize_and_padding():
    assert content_rect(640,480,256)==(0,.125,1,.875)
    assert inside([.4,.1,.6,.2],content_rect(640,480,256))
    assert not inside([.4,.01,.6,.03],content_rect(640,480,256))
    assert not inside([.4,.85,.6,.90],content_rect(640,480,256))


def test_human_diagnosis_separates_class_confusion_from_localization():
    entry=dict(ground_truth=[(1,.5,.5,.2,.2)],predictions=[(0,.9,[.4,.4,.6,.6])])
    result=human_overlap([entry])
    assert result['animal_only_overlap_at_50']==1
    assert result['human_overlap_at_50']==0
    entry['predictions'].append((1,.3,[.4,.4,.6,.6]))
    result=human_overlap([entry])
    assert result['human_overlap_at_50']==1
    assert result['animal_only_overlap_at_50']==0
