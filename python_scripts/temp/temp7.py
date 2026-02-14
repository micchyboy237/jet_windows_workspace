from wtpsplit import SaT

sat = SaT("sat-3l")
# optionally run on GPU for better performance
# also supports TPUs via e.g. sat.to("xla:0"), in that case pass `pad_last_batch=True` to sat.split
sat.half().to("cuda")

sat.split("This is a test This is another test.")
# returns ["This is a test ", "This is another test."]

# do this instead of calling sat.split on every text individually for much better performance
sat.split(["This is a test This is another test.", "And some more texts..."])
# returns an iterator yielding lists of sentences for every text

# use our '-sm' models for general sentence segmentation tasks
sat_sm = SaT("sat-3l-sm")
sat_sm.half().to("cuda") # optional, see above
sat_sm.split("this is a test this is another test")
# returns ["this is a test ", "this is another test"]

# use trained lora modules for strong adaptation to language & domain/style
sat_adapted = SaT("sat-3l", style_or_domain="ud", language="en")
sat_adapted.half().to("cuda") # optional, see above
sat_adapted.split("This is a test This is another test.")
# returns ['This is a test ', 'This is another test']