# QA Knowledge System

A comprehensive knowledge management system for QA (Quality Assurance) professionals.

## Features

- Knowledge base management
- Test case documentation
- QA process tracking
- Best practices repository

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository
```bash
git clone https://github.com/shamratneero/qa-knowledge-system.git
cd qa-knowledge-system
```

2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

The application expects an Excel knowledge base with the following columns:

- `id`, `question`, `answer`, `category`, `keywords`

By default the app will load `data/knowledge_base.xlsx`. You can also run the CLI and point it at any Excel file:

```bash
# run with the default file
python -m app.main

# run with a custom file path
python -m app.main --file path/to/your-file.xlsx
# or short form
python -m app.main -f path/to/your-file.xlsx
```

The loader will validate required columns and report errors if the file is missing or malformed.

## Notes

- If you have a spreadsheet named like `safeguest-support-tickets-2026-07-03.xlsx`, copy or move it into the `data/` directory and run the CLI with `--file` pointing at it, or rename it to `knowledge_base.xlsx` to use the default path.
- Development: create and activate a virtual environment before installing dependencies.

## Project Structure

```
qa-knowledge-system/
├── src/              # Main application source code
├── tests/            # Unit and integration tests
├── docs/             # Documentation
├── requirements.txt  # Project dependencies
└── README.md        # This file
```

## Development

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for development setup and guidelines.

## License

MIT
