from ai_arch_toolkit.toolkit.tools._filesystem import read_file, list_directory, search_files
from ai_arch_toolkit.toolkit.tools._json import csv_read
from ai_arch_toolkit.toolkit.tools._shell import run_command
for name, call in [("read_file", lambda: read_file("~no_such_user_zz/x")),
                   ("list_directory", lambda: list_directory("~no_such_user_zz")),
                   ("search_files", lambda: search_files("~no_such_user_zz", "a")),
                   ("csv_read", lambda: csv_read("~no_such_user_zz/x.csv"))]:
    try:
        call(); print(name, "ok?")
    except Exception as e:
        print(name, "->", type(e).__name__, str(e)[:70])
