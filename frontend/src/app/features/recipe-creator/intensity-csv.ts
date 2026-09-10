/** CSV reader supporting quoted fields, escaped quotes and spreadsheet delimiters. */
export function parseGroupCsv(text: string): string[][] {
  text = text.replace(/^\uFEFF/, '');
  const firstLine = text.split(/\r?\n/)[0];
  const counts = [',', ';', '\t'].map(separator => {
    let quoted = false, count = 0;
    for (const char of firstLine) {
      if (char === '"') quoted = !quoted;
      else if (!quoted && char === separator) count++;
    }
    return { separator, count };
  });
  const separator = counts.sort((a, b) => b.count - a.count)[0].separator;
  const rows: string[][] = [];
  let row: string[] = [], field = '', quoted = false, closed = false;
  const pushField = () => { row.push(field.trim()); field = ''; closed = false; };
  const pushRow = () => { pushField(); rows.push(row); row = []; };
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else { quoted = false; closed = true; }
      } else field += char;
    } else if (char === separator) pushField();
    else if (char === '\n' || char === '\r') {
      pushRow();
      if (char === '\r' && text[i + 1] === '\n') i++;
    } else if (char === '"' && !field.trim() && !closed) { field = ''; quoted = true; }
    else {
      if (char === '"' || (closed && char.trim())) throw new Error('Hibás CSV-idézőjelezés.');
      field += char;
    }
  }
  if (quoted) throw new Error('Lezáratlan idézőjel a CSV-ben.');
  if (field || row.length || closed) pushRow();
  if (!rows.length) throw new Error('A CSV üres.');
  return rows;
}

export function selectGroupLabels(rows: string[][], orientation: 'row' | 'column', index: number, skipFirst: boolean): string[] {
  if (!Number.isInteger(index) || index < 0) throw new Error('Érvénytelen sor- vagy oszlopszám.');
  const values = orientation === 'row' ? [...(rows[index] ?? [])] : rows.map(row => row[index] ?? '');
  if (skipFirst) values.shift();
  if (!values.length || values.some(value => !value.trim())) throw new Error('Minden mintához nem üres csoportnév szükséges.');
  return values.map(value => value.trim());
}
