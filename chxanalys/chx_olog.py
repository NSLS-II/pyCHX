from pyOlog import LogEntry,  Attachment,OlogClient
from pyOlog import SimpleOlogClient
from pyOlog.OlogDataTypes import  Logbook


def update_olog_id( logid, text, attachments):   
    '''Update olog book  logid entry with text and attachments files
    logid: integer, the log entry id
    text: the text to update, will add this text to the old text
    attachments: add new attachment files
    An example:
    
    filename1 = '/XF11ID/analysis/2016_2/yuzhang/Results/August/af8f66/Report_uid=af8f66.pdf'
    atch=[  Attachment(open(filename1, 'rb')) ] 
    
    update_olog_id( logid=29327, text='add_test_atch', attachmenents= atch )    
    
    '''
    url='https://logbook.nsls2.bnl.gov/Olog-11-ID/Olog'
    olog_client=SimpleOlogClient(  url= url, 
                                    username= 'xf11id', password= '***REMOVED***'   )
    
    client = OlogClient( url='https://logbook.nsls2.bnl.gov/Olog-11-ID/Olog', 
                                    username= 'xf11id', password= '***REMOVED***' )
    
    old_text =  olog_client.find( id = logid )[0]['text']    
    upd = LogEntry( text= old_text + '\n'+text,   attachments=  attachments,
                      logbooks= [Logbook( name = 'Operations', owner=None, active=True)]
                  )  
    upL = client.updateLog( logid, upd )    
    print( 'The url=%s was successfully updated with %s and with the attachments'%(url, text))
    
def update_olog_uid( uid, text, attachments):  
    '''Update olog book  logid entry cotaining uid string with text and attachments files
    uid: string, the uid of a scan or a specficial string (only gives one log entry)
    text: the text to update, will add this text to the old text
    attachments: add new attachment files    
    An example:
    
    filename1 = '/XF11ID/analysis/2016_2/yuzhang/Results/August/af8f66/Report_uid=af8f66.pdf'
    atch=[  Attachment(open(filename1, 'rb')) ] 
    update_olog_uid( uid='af8f66', text='Add xpcs pdf report', attachments= atch )    
    
    '''
    
    olog_client=SimpleOlogClient( url='https://logbook.nsls2.bnl.gov/Olog-11-ID/Olog', 
                                    username= 'xf11id', password= '***REMOVED***' )
    
    client = OlogClient( url='https://logbook.nsls2.bnl.gov/Olog-11-ID/Olog', 
                                    username= 'xf11id', password= '***REMOVED***' )
    
    logid = olog_client.find( search= uid )[0]['id']
    update_olog_id( logid, text, attachments)    
    