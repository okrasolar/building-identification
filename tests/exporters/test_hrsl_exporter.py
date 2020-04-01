from src.exporters import HRSLExporter


class TestHRSLExporter:
    def test_init(self, tmp_path):

        HRSLExporter(tmp_path, country_code="test")
        assert (tmp_path / "raw/hrsl_test_v1").exists()
