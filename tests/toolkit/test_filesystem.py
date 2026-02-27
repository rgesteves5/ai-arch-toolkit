"""Tests for toolkit/tools/_filesystem.py."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.tools._filesystem import list_directory, read_file, search_files


class TestReadFile:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = read_file(str(f))
        assert "line1" in result
        assert "line3" in result

    def test_truncation(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line {i}" for i in range(500)))
        result = read_file(str(f), max_lines=10)
        assert "Truncated" in result
        assert "500 total lines" in result

    def test_file_not_found(self):
        result = read_file("/nonexistent/path/file.txt")
        assert "not found" in result.lower()

    def test_directory_path(self, tmp_path):
        result = read_file(str(tmp_path))
        assert "Not a file" in result


class TestListDirectory:
    def test_list_files(self, tmp_path):
        (tmp_path / "a.py").write_text("code")
        (tmp_path / "b.txt").write_text("text")
        (tmp_path / "subdir").mkdir()
        result = list_directory(str(tmp_path))
        assert "a.py" in result
        assert "b.txt" in result
        assert "[dir]" in result
        assert "3 entries" in result

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("code")
        (tmp_path / "b.txt").write_text("text")
        result = list_directory(str(tmp_path), pattern="*.py")
        assert "a.py" in result
        assert "b.txt" not in result

    def test_nonexistent_dir(self):
        result = list_directory("/nonexistent/dir")
        assert "not found" in result.lower()

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = list_directory(str(f))
        assert "Not a directory" in result


class TestSearchFiles:
    def test_find_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    pass\n")
        (tmp_path / "b.py").write_text("def world():\n    pass\n")
        result = search_files(str(tmp_path), "hello")
        assert "a.py" in result
        assert "b.py" not in result

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "test.py").write_text("Hello World\n")
        result = search_files(str(tmp_path), "hello")
        assert "test.py" in result

    def test_no_matches(self, tmp_path):
        (tmp_path / "test.py").write_text("nothing here\n")
        result = search_files(str(tmp_path), "zzzzz")
        assert "No matches" in result

    def test_max_results(self, tmp_path):
        (tmp_path / "test.py").write_text("\n".join(f"match line {i}" for i in range(100)))
        result = search_files(str(tmp_path), "match", max_results=5)
        assert "Stopped at 5" in result

    def test_nonexistent_dir(self):
        result = search_files("/nonexistent", "pattern")
        assert "not found" in result.lower()
