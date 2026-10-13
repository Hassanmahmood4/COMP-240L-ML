const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  ImageRun,
  HeadingLevel,
  AlignmentType,
  PageBreak,
  BorderStyle,
  WidthType,
  Table,
  TableRow,
  TableCell,
  VerticalAlign,
} = require("docx");

const ROOT = path.join(__dirname, "..");

function readText(rel) {
  return fs.readFileSync(path.join(ROOT, rel), "utf8").replace(/\r\n/g, "\n");
}

function monoParagraphs(text, size = 16) {
  const lines = text.replace(/\t/g, "    ").split("\n");
  return lines.map(
    (line) =>
      new Paragraph({
        children: [
          new TextRun({
            text: line.length ? line : " ",
            font: "Courier New",
            size,
          }),
        ],
        spacing: { after: 0, line: 240 },
      })
  );
}

function sectionLabel(text) {
  return new Paragraph({
    spacing: { before: 240, after: 120 },
    children: [
      new TextRun({
        text,
        bold: true,
        size: 24,
        font: "Times New Roman",
      }),
    ],
  });
}

function taskHeading(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 200, after: 200 },
    children: [
      new TextRun({
        text,
        bold: true,
        size: 28,
        font: "Times New Roman",
      }),
    ],
  });
}

function taskStatement(text) {
  return new Paragraph({
    spacing: { after: 200 },
    children: [
      new TextRun({
        text,
        size: 22,
        font: "Times New Roman",
      }),
    ],
  });
}

function infoLine(label, value) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [
      new TextRun({
        text: `${label}: `,
        bold: true,
        size: 24,
        font: "Times New Roman",
      }),
      new TextRun({
        text: value,
        size: 24,
        font: "Times New Roman",
      }),
    ],
  });
}

function spacer(n = 1) {
  return Array.from(
    { length: n },
    () =>
      new Paragraph({
        children: [new TextRun({ text: " ", size: 22 })],
      })
  );
}

const tasks = [
  {
    title: "Task 1",
    statement:
      "Use PIP to install the numpy library. Verify the installation and import it into a Python script. Write a simple program to create and manipulate arrays using numpy functions like np.array() and np.sum().",
    codeFile: "Lab 2/task1_numpy.py",
    outputFile: "outputs/task1_output.txt",
  },
  {
    title: "Task 2",
    statement:
      "Install the requests library using PIP. Write a Python script that makes a GET request to a public API (e.g., weather data). Print the response content to the console and handle any potential exceptions gracefully.",
    codeFile: "Lab 2/task2_requests.py",
    outputFile: "outputs/task2_output.txt",
  },
  {
    title: "Task 3",
    statement:
      "Install beautifulsoup4 using PIP. Develop a Python script that scrapes a website for specific information (e.g., headlines or product prices). Use requests to fetch HTML content and beautifulsoup4 to parse and extract relevant data.",
    codeFile: "Lab 2/task3_beautifulsoup.py",
    outputFile: "outputs/task3_output.txt",
  },
  {
    title: "Task 4",
    statement:
      "Install pandas using PIP. Create a Python script that reads a CSV file containing sample data (Z-Alizadeh Sani dataset). Use pandas to load the data into a DataFrame, perform basic data manipulation (e.g., filtering, sorting), and generate summary statistics (e.g., mean, median).",
    codeFile: "Lab 2/task4_pandas.py",
    outputFile: "outputs/task4_output.txt",
  },
  {
    title: "Task 5",
    statement:
      "Set up a virtual environment using virtualenv or venv. Install scikit-learn, matplotlib, and pandas using PIP within the virtual environment. Develop a Python script that loads a dataset (e.g., Z-Alizadeh Sani dataset), preprocesses it using pandas, find the various rows and columns and visualizes the columns using matplotlib.",
    codeFile: "Lab 2/task5_visualization.py",
    outputFile: "outputs/task5_output.txt",
    imageFile: "outputs/task5_column_visualizations.png",
  },
];

const logoWide = fs.readFileSync(path.join(ROOT, "assets/pafiast_logo.png"));
const logoCircle = fs.readFileSync(path.join(ROOT, "assets/pafiast_logo_circle.png"));
const plotImg = fs.readFileSync(
  path.join(ROOT, "outputs/task5_column_visualizations.png")
);

const titlePage = [
  ...spacer(2),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [
      new ImageRun({
        type: "png",
        data: logoCircle,
        transformation: { width: 120, height: 120 },
        altText: { title: "PAF-IAST Logo", description: "University logo", name: "logo1" },
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 100 },
    children: [
      new ImageRun({
        type: "png",
        data: logoWide,
        transformation: { width: 420, height: 90 },
        altText: {
          title: "PAF-IAST",
          description: "Pak-Austria Fachhochschule logo",
          name: "logo2",
        },
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 80 },
    children: [
      new TextRun({
        text: "Pak-Austria Fachhochschule:",
        bold: true,
        size: 32,
        font: "Times New Roman",
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
    children: [
      new TextRun({
        text: "Institute of Applied Sciences and Technology (PAF-IAST)",
        bold: true,
        size: 26,
        font: "Times New Roman",
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 12, color: "1F4E79" },
    },
    spacing: { after: 300 },
    children: [
      new TextRun({
        text: "LAB REPORT",
        bold: true,
        size: 40,
        font: "Times New Roman",
        color: "1F4E79",
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
    children: [
      new TextRun({
        text: "Lab 06: Python Programming – PIP, Package Management,",
        bold: true,
        size: 26,
        font: "Times New Roman",
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
    children: [
      new TextRun({
        text: "and External Libraries in Python",
        bold: true,
        size: 26,
        font: "Times New Roman",
      }),
    ],
  }),
  infoLine("Course", "Machine Learning Lab"),
  infoLine("Department", "Artificial Intelligence"),
  infoLine("Semester", "________________"),
  ...spacer(1),
  infoLine("Submitted By", "________________"),
  infoLine("Registration No.", "________________"),
  infoLine("Section", "________________"),
  ...spacer(1),
  infoLine("Instructor Name", "________________"),
  infoLine("Lab Instructor", "________________"),
  ...spacer(1),
  infoLine("Submission Date", "07 October 2026"),
  new Paragraph({ children: [new PageBreak()] }),
];

const body = [];

for (const task of tasks) {
  body.push(taskHeading(task.title));
  body.push(taskStatement(task.statement));
  body.push(sectionLabel("Code"));
  body.push(...monoParagraphs(readText(task.codeFile), 15));
  body.push(sectionLabel("Output"));
  body.push(...monoParagraphs(readText(task.outputFile), 15));

  if (task.imageFile) {
    body.push(sectionLabel("Visualization Output"));
    body.push(
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 200 },
        children: [
          new ImageRun({
            type: "png",
            data: plotImg,
            transformation: { width: 520, height: 430 },
            altText: {
              title: "Task 5 Plot",
              description: "Column visualizations",
              name: "plot5",
            },
          }),
        ],
      })
    );
  }

  if (task !== tasks[tasks.length - 1]) {
    body.push(new Paragraph({ children: [new PageBreak()] }));
  }
}

const doc = new Document({
  styles: {
    default: {
      document: {
        styles: [
          {
            id: "Normal",
            run: { font: "Times New Roman", size: 22 },
          },
        ],
      },
    },
  },
  sections: [
    {
      properties: {
        page: {
          margin: { top: 720, bottom: 720, left: 864, right: 864 },
        },
      },
      children: [...titlePage, ...body],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  const reportsDir = path.join(ROOT, "Lab Reports");
  fs.mkdirSync(reportsDir, { recursive: true });
  const out = path.join(reportsDir, "ML LAB 2.docx");
  fs.writeFileSync(out, buffer);
  console.log("Created:", out);
});
