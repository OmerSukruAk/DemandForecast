import os
import img2pdf
import datetime as dt


def empty_directory(directory):
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))


def image_to_pdf(directory, output_pdf):

    image_files = [file for file in os.listdir(directory) if file.endswith((".jpg", ".jpeg", ".png"))]
    image_files.sort()

    output_pdf = output_pdf.replace(".pdf","") + str(str(dt.datetime.now()).split(" ")[0]) + ".pdf"

    image_paths = []

    for file in image_files:
        image_path = os.path.join(directory, file)
        image_paths.append(image_path)

    with open(output_pdf, "wb") as f:
        f.write(img2pdf.convert(image_paths)) # type: ignore

    empty_directory(directory)

    return output_pdf.split("/")[-1]


"""image_to_pdf("AnalysisCharts/stock_is_not_enough/", "reports/Ending_Stocks.pdf")
image_to_pdf("AnalysisCharts/stock_is_enough/", "reports/Enough_Stocks.pdf")"""