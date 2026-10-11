import re
import os
import argparse
import pickle
import random
from collections import defaultdict

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.python import PythonLexer

class CodeMarkovChain:
    def __init__(self, state_size=2):
        self.state_size = state_size
        self.model = defaultdict(list)

    def tokenize(self, text):
        return re.findall(r'\w+|[^\w\s]+|\s+', text)

    def train(self, text):
        tokens = self.tokenize(text)
        start_tokens = ['<START>'] * self.state_size
        tokens = start_tokens + tokens
        for i in range(len(tokens) - self.state_size):
            state = tuple(tokens[i : i + self.state_size])
            next_token = tokens[i + self.state_size]
            self.model[state].append(next_token)

    def train_on_directory(self, dir_path, extension):
        count = 0
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(extension):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            self.train(f.read())
                            count += 1
                    except Exception:
                        pass
        print(f"Trained on {count} files.")

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'state_size': self.state_size, 'model': self.model}, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.state_size = data['state_size']
            self.model = defaultdict(list, data['model'])

    def suggest(self, current_text, max_tokens=15):
        """Generates a short continuation based on the current text."""
        tokens = self.tokenize(current_text)
        if not tokens:
            return ""

        # Pad with <START> if the user just started typing
        if len(tokens) < self.state_size:
            padding = ['<START>'] * (self.state_size - len(tokens))
            current_state = tuple(padding + tokens)
        else:
            current_state = tuple(tokens[-self.state_size:])

        output = []
        open_brackets = []

        for _ in range(max_tokens):
            choices = self.model.get(current_state)
            
            # Katz Backoff
            backoff_state = current_state
            while not choices and len(backoff_state) > 0:
                backoff_state = backoff_state[1:]
                choices = self.model.get(backoff_state)
                
            if not choices:
                break
                
            next_token = random.choice(choices)
            if next_token == '<START>':
                continue
                
            output.append(next_token)
            
            if next_token in '([{': 
                open_brackets.append(next_token)
            elif next_token in ')]}':
                if open_brackets: open_brackets.pop()
            
            current_state = tuple(list(current_state[1:]) + [next_token])

        # Auto-close brackets
        bracket_map = {'(': ')', '[': ']', '{': '}'}
        while open_brackets:
            output.append(bracket_map[open_brackets.pop()])

        return ''.join(output)


class MarkovAutoSuggest(AutoSuggest):
    """Custom AutoSuggest class for prompt_toolkit"""
    def __init__(self, markov_chain):
        self.chain = markov_chain
        self.last_text = ""
        self.last_suggestion = ""

    def get_suggestion(self, buffer, document):
        current_text = document.text_before_cursor
        
        # 1. If the user just typed the exact next character of the last suggestion, 
        # keep the rest of the suggestion to avoid flickering/re-generating.
        if (current_text.startswith(self.last_text) and 
            len(current_text) == len(self.last_text) + 1 and 
            self.last_suggestion.startswith(current_text[-1])):
            
            self.last_text = current_text
            self.last_suggestion = self.last_suggestion[1:]
            return Suggestion(self.last_suggestion)

        # 2. If the user deletes text, or types something different, clear old state
        if not current_text.startswith(self.last_text):
            self.last_text = ""
            self.last_suggestion = ""

        # 3. Don't suggest on empty lines or immediately after pressing enter
        if not current_text.strip() or current_text.endswith('\n'):
            return Suggestion("")

        # 4. Generate a new suggestion
        new_suggestion = self.chain.suggest(current_text)
        
        # Update state so we can track if they accept it character-by-character
        self.last_text = current_text
        self.last_suggestion = new_suggestion
        
        return Suggestion(new_suggestion)


def main():
    parser = argparse.ArgumentParser(description="Interactive Markov Code Autocomplete")
    parser.add_argument('--train', type=str, help="Directory to train on before starting")
    parser.add_argument('--ext', type=str, default='.py', help="File extension to train on")
    parser.add_argument('--load', type=str, help="Load a saved .pkl model")
    parser.add_argument('--state', type=int, default=2, help="State size (N-gram)")
    
    args = parser.parse_args()
    chain = CodeMarkovChain(state_size=args.state)

    if args.load:
        chain.load_model(args.load)
        print(f"Loaded model. {len(chain.model)} states.")
    elif args.train:
        print(f"Training on {args.train}...")
        chain.train_on_directory(args.train, args.ext)
        print(f"Training complete. {len(chain.model)} states.")
    else:
        print("Error: Please provide --train <dir> or --load <model.pkl>")
        return

    print("\nStarting Interactive Code Editor...")
    print("Type your code. Suggestions will appear in gray.")
    print("Press [TAB] to accept the suggestion.")
    print("Press [Ctrl+C] to exit.\n")

    # Create custom keybindings
    kb = KeyBindings()

    @kb.add('tab')
    def _(event):
        """When Tab is pressed, accept the auto-suggestion."""
        buffer = event.app.current_buffer
        suggestion = buffer.suggestion
        if suggestion and suggestion.text:
            buffer.insert_text(suggestion.text)
        else:
            # If no suggestion, insert a normal tab character
            buffer.insert_text('\t')

    # Set up the prompt session
    session = PromptSession(
        lexer=PygmentsLexer(PythonLexer),  # Syntax highlighting
        auto_suggest=MarkovAutoSuggest(chain),  # Our Markov ghost text
        key_bindings=kb,
        multiline=True,  # Allow pressing Enter without submitting the prompt
        prompt_continuation='... '  # Shows ... on new lines
    )

    try:
        # This runs the interactive loop until Ctrl+C is pressed
        text = session.prompt('> ')
    except KeyboardInterrupt:
        print("\nExiting editor. Goodbye!")

if __name__ == "__main__":
    main()