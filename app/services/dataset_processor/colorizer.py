from bs4 import BeautifulSoup

GREEN_COLOR = '#2ed94a'
ORANGE_COLOR = '#f2931f'
BLUE_COLOR = '#2a3de8'


def colorize_html_table(
    matched_indices: list[tuple[int, int]],
    false_positive_indices: list[tuple[int, int]],
    false_negative_indices: list[tuple[int, int]],
    html_table_content: str
):
    bs = BeautifulSoup(html_table_content, features='html.parser')
    cells = [
        row.find_all('td', recursive=False) for row in bs.find('tbody').find_all('tr', recursive=False)
    ]
    
    try:
        for row, col in matched_indices:
            cells[row][col]['bgcolor'] = GREEN_COLOR

        for row, col in false_positive_indices:
            cells[row][col]['bgcolor'] = ORANGE_COLOR

        for row, col in false_negative_indices:
            cells[row][col]['bgcolor'] = BLUE_COLOR
    except IndexError:
        pass
    
    return str(bs)
