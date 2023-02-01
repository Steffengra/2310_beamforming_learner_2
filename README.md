
# Under Construction


#### Libraries you should use:
- logging
- unittests


#### Folder Structure

|                  |                                                  |
|------------------|--------------------------------------------------|
| data             |                                                  |
| ├─ interim       | intermediate data                                |
| ├─ raw           | input data, treat as immutable                   |
| └─ processed     | final data sets for modeling                     |
| models           | trained models or summaries                      |
| outputs          | outputs produced by src                          |
| references       | manuals etc.                                     |
| reports          | generated analysis, latex etc.                   |
| └─ figures       |                                                  |
| src              |                                                  |
| ├─ analysis      | results oriented exploration                     |
| ├─ config        | configuration files                              |
| ├─ data          | data generation, loading, etc.                   |
| ├─ models        | model training and inference                     |
| ├─ notebooks     | sketching                                        |
| ├─ tests         | unit tests etc.                                  |
| └─ utils         | functions shared between different parts of code |
| .gitignore       | .gitignore                                       |
| README.md        | readme                                           |
| requirements.txt | pip freeze > requirements.txt                    |

#### Markdown
|            |                                  |
|------------|----------------------------------|
| Heading    | #, ##, ###, ...                  |
| Bold       | `**bold**`                       |
| Italic     | `*italic*`                       |
| Blockquote | `> blockquote`                   |
| Enumerate  | 1. a<br/>2. b<br/>...            |
| Itemize    | - a<br/>- b<br/>...              |
| code       | `code`                           |
| Hrule      | ---                              |
| Link       | [title](https://www.example.com) |
