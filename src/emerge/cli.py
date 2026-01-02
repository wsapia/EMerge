import argparse
from ._emerge.projects.generate_project import generate_project

def main():
    parser = argparse.ArgumentParser(description="Emerge Project Generator CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand: new
    new_parser = subparsers.add_parser("new", help="Create a new project")
    new_parser.add_argument("projectname", type=str, help="Name of the project directory")
    new_parser.add_argument("filename", type=str, help="Base name for files")

    args = parser.parse_args()

    if args.command == "new":
        generate_project(args.projectname, args.filename)
    else:
        parser.print_help()
