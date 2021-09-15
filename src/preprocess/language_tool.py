import os
import language_tool_python


def correct(text, language):
    language_tool = language_tool_python.LanguageTool(language)
    correct_text = language_tool.correct(text)
    return correct_text


def main():
    language_tool = language_tool_python.LanguageTool('es')
    correct_text = language_tool.correct("El paro ha bajado en España un 10% en los últimos 5 días.")
    print(correct_text)


if __name__ == '__main__':
    main()