import { parseGroupCsv, selectGroupLabels } from './intensity-csv';

describe('Intensity group CSV', () => {
  it('imports a BOM-prefixed row with quoted delimiters and escaped quotes', () => {
    expect(selectGroupLabels(parseGroupCsv('\uFEFF"A,1","B""2",C\r\n'), 'row', 0, false)).toEqual(['A,1', 'B"2', 'C']);
  });
  it('selects a semicolon-separated column and skips its header', () => {
    expect(selectGroupLabels(parseGroupCsv('Minta;Csoport\r\n1;A\r\n2;B\r\n'), 'column', 1, true)).toEqual(['A', 'B']);
  });
  it('selects a tab-separated row and skips its label', () => {
    expect(selectGroupLabels(parseGroupCsv('képek\t1\t2\ncsoport\tA\tA'), 'row', 1, true)).toEqual(['A', 'A']);
  });
  it('rejects empty entries without shifting the image mapping', () => {
    expect(() => selectGroupLabels(parseGroupCsv('A,,B'), 'row', 0, false)).toThrow();
    expect(() => selectGroupLabels(parseGroupCsv('A\n\nB'), 'column', 0, false)).toThrow();
    expect(() => selectGroupLabels([['A']], 'column', 2, false)).toThrow();
    expect(() => parseGroupCsv('"unclosed')).toThrow();
  });
});
