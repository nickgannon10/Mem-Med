import modal 
import os

stub = modal.Stub()

@stub.local_entrypoint
def main(json_path): 
    import json
    with open(json_path) as f:
        pdf_infos = json.load(f)
    
    pdf_urls = [pdf["url"] for pdf in pdf_infos]

    results = list(extract_pdf.map(pdf_urls, return_exceptions=True))
    add_to_document_db.call(results)






# @stub.function(
#     secret=modal.Secret.from_name("some_secret"),
#     schedule=modal.Period(days=1),
# )
# def foo():
#     pass

# @stub.local_entrypoint()
# def main():
#     output_dir = "/tmp/nyc"
#     os.makedirs(output_dir, exist_ok=True)

#     fn = os.path.join(output_dir, "nyc_taxi_chart.png")

#     with stub.run():
#         png_data = create_plot.call()
#         with open(fn, "wb") as f:
#             f.write(png_data)
#         print(f"wrote output to {fn}")