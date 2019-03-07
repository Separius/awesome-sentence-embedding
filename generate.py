import json
import urllib.parse
import urllib.request
from operator import attrgetter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

arxiv_prefix = 'https://arxiv.org/abs/'
arxiv_prefix_len = len(arxiv_prefix)
github_prefix = 'https://github.com/'
github_prefix_len = len(github_prefix)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def fancy_code(code):
    return '[{training_language}]({code_link} ){unofficial}'.format(
        training_language=code['training_language'], code_link=code['code_link'],
        unofficial='(unofficial)' if code.get('unofficial', False) else '')


def query_semantic_scholar(query):
    if query == '':
        return 'N/A', '-'
    try:
        res = json.loads(urllib.request.urlopen("https://api.semanticscholar.org/v1/paper/" + query).read())
        count = len(res['citations'])
        return (str(count) if count < 999 else '999+'), str(res['year']) + '/??'
    except:
        return 'N/A', '-'


def fetch_common_parts(paper):
    is_arxiv = paper['paper_link'].startswith(arxiv_prefix)
    if is_arxiv:
        arxiv_id = paper['paper_link'][arxiv_prefix_len:]
        arxiv_date = arxiv_id.split('.')[0]
        date_part = '20' + arxiv_date[:2] + '/' + arxiv_date[2:]
        citation_part, _ = query_semantic_scholar('arXiv:{}'.format(arxiv_id))
    else:
        citation_part, date_part = query_semantic_scholar(paper.get('doi', paper.get('s2_paper_id', '')))
    paper_part = '[{paper_title}]({paper_link})'.format(paper_title=paper['paper_title'],
                                                        paper_link=paper['paper_link'])
    return citation_part, date_part, paper_part


def generate_word_embedding_table():
    header = ['|date|paper|citation count|training code|pretrained models|',
              '|:---:|:---:|:---:|:---:|:---:|']
    generated_lines = []
    with open('word-embedding.json') as f:
        meta_info = json.load(f)
    for paper in tqdm(meta_info):
        citation_part, date_part, paper_part = fetch_common_parts(paper)
        if 'code' in paper:
            training_code_part = fancy_code(paper['code'][0])
        else:
            training_code_part = '-'
        pretrained_part = '-' if 'name' not in paper else '[{name}]({pretrained_link} ){broken_link}' \
            .format(name=paper['name'], pretrained_link=paper['pretrained_link'],
                    broken_link='(broken)' if paper.get('broken_link', False) else '')
        generated_lines.append(
            AttrDict(date_part=date_part, paper_part=paper_part,
                     training_code_part=training_code_part,
                     pretrained_part=pretrained_part, citation_part=citation_part))
    generated_lines = sorted(generated_lines, key=attrgetter('date_part', 'citation_part'))
    generated_lines = [
        '|{date_part}|{paper_part}|{citation_part}|{training_code_part}|{pretrained_part}|'.format(**x) for x in
        generated_lines]
    return '\n'.join(header + generated_lines)


def generate_contextualized_table():
    header = ['|date|paper|citation count|code|pretrained models|',
              '|:---:|:---:|:---:|:---:|:---:|']
    generated_lines = []
    with open('contextualized.json') as f:
        meta_info = json.load(f)
    for paper in tqdm(meta_info):
        citation_part, date_part, paper_part = fetch_common_parts(paper)
        if 'code' in paper:
            training_code_part = '<br>'.join([fancy_code(code) for code in paper['code']])
        else:
            training_code_part = '-'
        if 'pretrained_models' in paper:
            if len(paper['pretrained_models']) == 1:
                pretrained_models = '[{name}]({pretrained_link} )'.format(name=paper['model_name'],
                                                                          pretrained_link=paper['pretrained_models'][0][
                                                                              'link'])
            else:
                pretrained_links = ', '.join(
                    ['[{name}](link)'.format(name=x['name'], link=x['link']) for x in paper['pretrained_models']])
                pretrained_models = '{name}({pretrained_link})'.format(name=paper['model_name'],
                                                                       pretrained_link=pretrained_links)
        else:
            pretrained_models = '-'
        generated_lines.append(
            AttrDict(date_part=date_part, paper_part=paper_part, training_code_part=training_code_part,
                     pretrained_models=pretrained_models, citation_part=citation_part))
    generated_lines = sorted(generated_lines, key=attrgetter('date_part', 'citation_part'))
    generated_lines = [
        '|{date_part}|{paper_part}|{citation_part}|{training_code_part}|{pretrained_models}|'.format(**x) for x in
        generated_lines]
    return '\n'.join(header + generated_lines)

#unoficial, official, pretrained, load_pretrained, email_for_pretrained, no_training_code

if __name__ == '__main__':
    with open('README_BASE.md') as f:
        readme = f.read()
    readme = readme.replace('{{{word-embedding-table}}}', generate_word_embedding_table())
    readme = readme.replace('{{{contextualized-table}}}', generate_contextualized_table())
    # readme = readme.replace('{{{encoder-table}}}', generate_encoder_table())
    with open('README.md', 'w') as f:
        f.write(readme)
