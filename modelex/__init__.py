import argparse
import importlib
import sys

def main():
    # top-level parser
    parser = argparse.ArgumentParser(description="Modelex CLI")
    parser.set_defaults(func=lambda _: parser.print_help())
    parser.add_argument('subcommand', type=str, nargs='?', help='{train,serve,prompt,generate,prepare_dataset,benchmark}')
    args, unknown = parser.parse_known_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    try:
        module = importlib.import_module(f'modelex.scripts.{args.subcommand}')
        if not hasattr(module, 'main'):
            print(f"Error: {args.subcommand} does not have a main() function")
            sys.exit(1)
        if hasattr(module, 'parser'):
            module.main(args=module.parser.parse_args(unknown))
        else:
            module.main()

    except ImportError as e:
        print(f"Error: {args.subcommand} not found\n", e)
        sys.exit(1)
    except Exception as e:
        print(f"Error running {args.subcommand}: {e}")
        raise e

if __name__ == '__main__':
    main()