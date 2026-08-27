def test_public_classes_importable_from_package_root():
    from turboquant import PolarQuant, TurboQuantMSE, TurboQuantProd

    assert TurboQuantMSE is not None
    assert TurboQuantProd is not None
    assert PolarQuant is not None
